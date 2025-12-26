import asyncio
import hashlib
import json
import os
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import yaml
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType

from whorl.lib import text_edit

load_dotenv()

WHORL_DIR = Path.home() / ".whorl"
SETTINGS_PATH = WHORL_DIR / "settings.json"

WHORL_API_KEY = os.environ.get("WHORL_API_KEY")

router = APIRouter()

_settings: dict | None = None


def load_settings() -> dict:
    global _settings
    if _settings is None:
        with open(SETTINGS_PATH) as f:
            _settings = json.load(f)
    return _settings


def get_docs_dir() -> Path:
    settings = load_settings()
    docs_path = settings.get("docs_dir", "docs")
    if Path(docs_path).is_absolute():
        return Path(docs_path)
    return WHORL_DIR / docs_path


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


HASH_INDEX_PATH = WHORL_DIR / "hash-index.json"


def load_hash_index() -> dict[str, dict]:
    """Load the hash index from disk."""
    if HASH_INDEX_PATH.exists():
        with open(HASH_INDEX_PATH) as f:
            return json.load(f)
    return {}


def save_hash_index(index: dict[str, dict]) -> None:
    """Save the hash index to disk."""
    with open(HASH_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)


def add_to_hash_index(content_hash: str, doc_id: str, path: str) -> None:
    """Add a document to the hash index."""
    index = load_hash_index()
    index[content_hash] = {"id": doc_id, "path": path}
    save_hash_index(index)


def find_doc_by_hash(content_hash: str) -> dict | None:
    """Find an existing doc by content hash. O(1) lookup."""
    index = load_hash_index()
    return index.get(content_hash)


def remove_from_hash_index(content_hash: str) -> None:
    """Remove a document from the hash index."""
    index = load_hash_index()
    if content_hash in index:
        del index[content_hash]
        save_hash_index(index)


class IngestRequest(BaseModel):
    content: str
    metadata: dict[str, Any] | None = None
    title: str | None = None
    process: bool = False


class IngestResponse(BaseModel):
    id: str
    path: str
    duplicate: bool = False
    content_hash: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    context: int = 2


class BashRequest(BaseModel):
    command: str
    timeout: int = 30


class BashResponse(BaseModel):
    stdout: str
    stderr: str
    returncode: int


class DeleteRequest(BaseModel):
    path: str


class ReingestRequest(BaseModel):
    path: str


class UpdateRequest(BaseModel):
    path: str
    content: str
    title: str | None = None


class SearchResult(BaseModel):
    id: str
    path: str
    title: str | None
    snippet: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int


class AgentSearchRequest(BaseModel):
    query: str


class AgentSearchResponse(BaseModel):
    answer: str


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(api_key_header)):
    """Verify API key from header."""
    if not WHORL_API_KEY:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if not api_key or api_key != WHORL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
                return frontmatter, body
            except yaml.YAMLError:
                pass
    return {}, content


def search_docs(query_str: str, limit: int = 10, context: int = 2, exclude: list[str] | None = None) -> list[SearchResult]:
    """Full text search using ripgrep."""
    docs_dir = get_docs_dir()
    if not docs_dir.exists():
        return []

    cmd = [
        "rg",
        "--json",
        "-i",  # case insensitive
        "-C", str(context),  # context lines
    ]

    # Add exclude patterns
    for pattern in (exclude or []):
        cmd.extend(["--glob", f"!{pattern}"])

    cmd.extend([query_str, str(docs_dir)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    # Parse ripgrep JSON output
    matches: dict[str, dict] = {}

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if data["type"] == "match":
            path = data["data"]["path"]["text"]
            line_text = data["data"]["lines"]["text"].strip()
            line_num = data["data"]["line_number"]

            if path not in matches:
                matches[path] = {"lines": [], "match_count": 0}
            matches[path]["lines"].append((line_num, line_text))
            matches[path]["match_count"] += 1

        elif data["type"] == "context":
            path = data["data"]["path"]["text"]
            line_text = data["data"]["lines"]["text"].strip()
            line_num = data["data"]["line_number"]

            if path not in matches:
                matches[path] = {"lines": [], "match_count": 0}
            matches[path]["lines"].append((line_num, line_text))

    # Build results
    results = []
    for filepath_str, match_data in matches.items():
        filepath = Path(filepath_str)
        content = filepath.read_text()
        frontmatter, _ = parse_frontmatter(content)

        # Sort lines by line number and build snippet
        match_data["lines"].sort(key=lambda x: x[0])
        snippet = "\n".join(line for _, line in match_data["lines"])

        results.append(SearchResult(
            id=frontmatter.get("id", filepath.stem),
            path=str(filepath),
            title=frontmatter.get("title"),
            snippet=snippet,
            score=float(match_data["match_count"]),
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]


ANTHROPIC_TOOLS = [
    {"type": "text_editor_20250124", "name": "str_replace_editor"},
    {"type": "bash_20250124", "name": "bash"},
]

MODEL_MAP = {
    "haiku": "claude-sonnet-4-20250514",  # Use sonnet as haiku replacement for tool use
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
}


def run_bash_command(command: str, cwd: str, timeout: int = 30) -> tuple[str, str, int]:
    """Execute a bash command and return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", -1
    except Exception as e:
        return "", str(e), -1


def format_bash_output(stdout: str, stderr: str, returncode: int) -> str:
    """Format bash output for agent consumption."""
    output = stdout
    if stderr:
        output += f"\n[stderr]: {stderr}" if output else f"[stderr]: {stderr}"
    if returncode != 0:
        output += f"\n[exit code: {returncode}]"
    return output or "(no output)"


def truncate_if_huge(output: str, limit: int = 100000) -> str:
    """Only truncate if output is very large (100k+ chars)."""
    if len(output) <= limit:
        return output
    return output[:limit] + f"\n\n[truncated - was {len(output)} chars]"


def execute_tool(name: str, inputs: dict, cwd: str) -> str:
    """Execute a tool and return the result (truncated if huge)."""
    if name == "bash":
        command = inputs.get("command", "")
        print(f"  [bash] {command[:80]}{'...' if len(command) > 80 else ''}")
        stdout, stderr, returncode = run_bash_command(command, cwd)
        return truncate_if_huge(format_bash_output(stdout, stderr, returncode))

    elif name == "str_replace_editor":
        command = inputs.get("command", "")
        path = inputs.get("path", "")
        # Make path absolute relative to cwd
        if path and not path.startswith("/"):
            path = str(Path(cwd) / path)
        # Security: ensure path stays within cwd
        try:
            resolved = Path(path).resolve()
            cwd_resolved = Path(cwd).resolve()
            if not str(resolved).startswith(str(cwd_resolved)):
                return f"Error: path {path} is outside allowed directory"
        except Exception:
            return f"Error: invalid path {path}"
        print(f"  [edit] {command} {path}")
        # Remove command/path from inputs since we're passing them explicitly
        kwargs = {k: v for k, v in inputs.items() if k not in ("command", "path")}
        return truncate_if_huge(text_edit.execute(command, path, **kwargs))

    return f"Unknown tool: {name}"


async def run_agent_loop(
    client: anthropic.AsyncAnthropic,
    system_prompt: str,
    user_message: str,
    model: str,
    max_turns: int,
    cwd: str,
) -> str:
    """Run a multi-turn agent conversation with bash and text_editor tools."""
    model_id = MODEL_MAP.get(model, model)
    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = await client.beta.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system_prompt,
            tools=ANTHROPIC_TOOLS,
            messages=messages,
            betas=["computer-use-2025-01-24"],
        )

        # Check if we're done (no tool use)
        has_tool_use = any(block.type == "tool_use" for block in response.content)

        if not has_tool_use:
            # Extract final text response
            text_parts = [block.text for block in response.content if block.type == "text"]
            return "\n".join(text_parts) if text_parts else ""

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input, cwd)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn":
            break

    return ""


async def run_single_ingestion_agent(content: str, filepath: Path, prompt_path: Path, config: dict) -> str | None:
    """Run a single ingestion agent with a specific prompt. Returns the prompt name or None on failure."""
    try:
        prompt_raw = prompt_path.read_text()
        prompt_frontmatter, prompt_body = parse_frontmatter(prompt_raw)
        system_prompt = prompt_body.format(filepath=filepath)

        # Config from frontmatter, fall back to global config
        model = prompt_frontmatter.get("model", config.get("model", "sonnet"))
        max_turns = prompt_frontmatter.get("max_turns", config.get("max_turns", 50))

        client = anthropic.AsyncAnthropic()

        await run_agent_loop(
            client=client,
            system_prompt=system_prompt,
            user_message=content,
            model=model,
            max_turns=max_turns,
            cwd=str(get_docs_dir()),
        )

        return prompt_path.stem
    except Exception as e:
        import traceback
        print(f"Agent {prompt_path.stem} failed: {e}")
        traceback.print_exc()
        return None


async def run_ingestion_agent(content: str, filepath: Path, settings: dict) -> None:
    """Run all ingestion agents in parallel."""
    if not content or not content.strip():
        return

    config = settings["ingestion_config"]
    prompts_dir = WHORL_DIR / config["prompts_dir"]

    if not prompts_dir.exists():
        return

    prompt_files = list(prompts_dir.glob("*.md"))
    if not prompt_files:
        return

    tasks = [
        run_single_ingestion_agent(content, filepath, prompt_path, config)
        for prompt_path in prompt_files
    ]
    results = await asyncio.gather(*tasks)
    agent_names = [name for name in results if name is not None]

    # Update frontmatter with processed agents
    file_content = filepath.read_text()
    frontmatter, body = parse_frontmatter(file_content)
    frontmatter["processed"] = sorted(agent_names)
    yaml_frontmatter = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    filepath.write_text(f"---\n{yaml_frontmatter}---\n\n{body}")


async def run_search_agent(query_str: str, settings: dict) -> str:
    config = settings["search_config"]
    prompt_path = WHORL_DIR / config["prompt"]
    prompt_raw = prompt_path.read_text()
    prompt_frontmatter, prompt_body = parse_frontmatter(prompt_raw)
    docs_dir = get_docs_dir()
    system_prompt = prompt_body.format(docs_dir=docs_dir)

    # Config from frontmatter, fall back to global config
    model = prompt_frontmatter.get("model", config.get("model", "sonnet"))
    max_turns = prompt_frontmatter.get("max_turns", config.get("max_turns", 25))

    client = anthropic.AsyncAnthropic()

    result = await run_agent_loop(
        client=client,
        system_prompt=system_prompt,
        user_message=query_str,
        model=model,
        max_turns=max_turns,
        cwd=str(docs_dir),
    )

    return result or "No results found."


@router.post("/ingest", response_model=IngestResponse, tags=["mcp"])
async def ingest(request: IngestRequest, _: None = Depends(verify_api_key)):
    """Ingest content into the knowledge base."""
    docs_dir = get_docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings()

    # Compute content hash and check for duplicates
    content_hash = compute_content_hash(request.content)
    existing = find_doc_by_hash(content_hash)

    if existing:
        return IngestResponse(
            id=existing["id"],
            path=existing["path"],
            duplicate=True,
            content_hash=content_hash,
        )

    doc_id = os.urandom(4).hex()
    timestamp = datetime.now(timezone.utc).isoformat()

    if request.title:
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in request.title)
        safe_title = safe_title.strip().replace(" ", "-")[:50]
        filename = f"{safe_title}-{doc_id}.md" if safe_title else f"{doc_id}.md"
    else:
        filename = f"{doc_id}.md"

    filepath = docs_dir / filename

    frontmatter = {
        "id": doc_id,
        "created_at": timestamp,
        "content_hash": content_hash,
    }
    if request.title:
        frontmatter["title"] = request.title
    if request.metadata:
        frontmatter.update(request.metadata)

    yaml_frontmatter = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    doc_content = f"---\n{yaml_frontmatter}---\n\n{request.content}"
    filepath.write_text(doc_content)

    # Add to hash index
    add_to_hash_index(content_hash, doc_id, filepath.name)

    if request.process:
        await run_ingestion_agent(request.content, filepath, settings)

    return IngestResponse(id=doc_id, path=filepath.name, content_hash=content_hash)


@router.post("/search", response_model=SearchResponse, tags=["mcp"])
async def search(request: SearchRequest, _: None = Depends(verify_api_key)):
    """Full text search over the knowledge base."""
    settings = load_settings()
    exclude = settings.get("search_config", {}).get("exclude", [])
    results = search_docs(request.query, request.limit, request.context, exclude)
    return SearchResponse(results=results, total=len(results))


@router.post("/agent_search", response_model=AgentSearchResponse, tags=["mcp"])
async def agent_search(request: AgentSearchRequest, _: None = Depends(verify_api_key)):
    """Agent-powered search over the knowledge base."""
    settings = load_settings()
    answer = await run_search_agent(request.query, settings)
    return AgentSearchResponse(answer=answer)


@router.post("/bash", response_model=BashResponse, tags=["mcp"])
async def bash(request: BashRequest, _: None = Depends(verify_api_key)):
    """Run a bash command in the docs directory."""
    docs_dir = get_docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)

    stdout, stderr, returncode = run_bash_command(request.command, str(docs_dir), request.timeout)
    if returncode == -1 and stderr == "Command timed out":
        raise HTTPException(status_code=408, detail="Command timed out")
    return BashResponse(stdout=stdout, stderr=stderr, returncode=returncode)


@router.post("/delete")
async def delete_doc(request: DeleteRequest, _: None = Depends(verify_api_key)):
    """Delete a document."""
    docs_dir = get_docs_dir()
    filepath = docs_dir / request.path

    # Security: ensure path is within docs_dir
    try:
        filepath = filepath.resolve()
        docs_dir.resolve()
        if not str(filepath).startswith(str(docs_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid path")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from hash index
    content = filepath.read_text()
    frontmatter, _ = parse_frontmatter(content)
    if content_hash := frontmatter.get("content_hash"):
        remove_from_hash_index(content_hash)

    filepath.unlink()
    return {"status": "deleted", "path": request.path}


@router.post("/update")
async def update_doc(request: UpdateRequest, _: None = Depends(verify_api_key)):
    """Update an existing document."""
    docs_dir = get_docs_dir()
    filepath = docs_dir / request.path

    # Security: ensure path is within docs_dir
    try:
        filepath = filepath.resolve()
        if not str(filepath).startswith(str(docs_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid path")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    # Read existing frontmatter
    existing_content = filepath.read_text()
    frontmatter, _ = parse_frontmatter(existing_content)

    # Update title if provided
    if request.title is not None:
        frontmatter["title"] = request.title

    yaml_frontmatter = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    doc_content = f"---\n{yaml_frontmatter}---\n\n{request.content}"
    filepath.write_text(doc_content)

    return {"status": "updated", "path": request.path}


class ReingestResponse(BaseModel):
    status: str
    path: str
    processed: list[str]
    skipped: bool = False


@router.post("/reingest", response_model=ReingestResponse)
async def reingest_doc(request: ReingestRequest, _: None = Depends(verify_api_key)):
    """Re-run ingestion agents on an existing document (only missing agents)."""
    docs_dir = get_docs_dir()
    filepath = docs_dir / request.path

    # Security check
    try:
        filepath = filepath.resolve()
        if not str(filepath).startswith(str(docs_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid path")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    content = filepath.read_text()
    frontmatter, body = parse_frontmatter(content)

    settings = load_settings()
    config = settings.get("ingestion_config", {})
    prompts_dir = WHORL_DIR / config.get("prompts_dir", "prompts/ingestion")

    # Get expected vs processed agents
    expected_agents = {p.stem for p in prompts_dir.glob("*.md")} if prompts_dir.exists() else set()
    processed_agents = set(frontmatter.get("processed", []))
    missing_agents = expected_agents - processed_agents

    if not missing_agents:
        return ReingestResponse(status="skipped", path=request.path, processed=list(processed_agents), skipped=True)

    # Run only missing agents
    prompt_files = [prompts_dir / f"{agent}.md" for agent in missing_agents]
    tasks = [
        run_single_ingestion_agent(body, filepath, prompt_path, config)
        for prompt_path in prompt_files if prompt_path.exists()
    ]
    results = await asyncio.gather(*tasks)
    new_agents = [name for name in results if name is not None]

    # Update frontmatter with all processed agents
    all_processed = sorted(processed_agents | set(new_agents))
    file_content = filepath.read_text()
    frontmatter, body = parse_frontmatter(file_content)
    frontmatter["processed"] = all_processed
    yaml_frontmatter = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    filepath.write_text(f"---\n{yaml_frontmatter}---\n\n{body}")

    return ReingestResponse(status="reingested", path=request.path, processed=all_processed)


@router.get("/settings")
async def get_settings(_: None = Depends(verify_api_key)):
    """Get current settings."""
    return load_settings()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/documents")
async def list_docs(_: None = Depends(verify_api_key)):
    """List all documents (recursive)."""
    docs_dir = get_docs_dir()
    docs = []
    for filepath in docs_dir.glob("**/*.md"):
        content = filepath.read_text()
        frontmatter, body = parse_frontmatter(content)
        # Use relative path from docs_dir
        rel_path = filepath.relative_to(docs_dir)
        docs.append({
            "id": frontmatter.get("id", filepath.stem),
            "path": str(rel_path),
            "title": frontmatter.get("title"),
            "created_at": frontmatter.get("created_at"),
            "frontmatter": frontmatter,
        })
    docs.sort(key=lambda d: d.get("created_at") or "", reverse=True)
    return {"docs": docs}


@router.get("/documents/{path:path}")
async def get_doc(path: str, _: None = Depends(verify_api_key)):
    """Get a document's content."""
    docs_dir = get_docs_dir()
    filepath = docs_dir / path

    # Security check
    if not str(filepath.resolve()).startswith(str(docs_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Not found")

    return {"content": filepath.read_text()}


@router.get("/download/{path:path}")
async def download_file(
    path: str,
    api_key: str | None = None,
    header_key: str | None = Depends(api_key_header),
):
    """Download a file from the docs directory (PDFs, etc)."""
    # Accept API key from either query param or header (for browser downloads)
    key = api_key or header_key
    if not WHORL_API_KEY:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if not key or key != WHORL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    docs_dir = get_docs_dir()
    filepath = docs_dir / path

    # Security check
    if not str(filepath.resolve()).startswith(str(docs_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(
        filepath,
        filename=filepath.name,
        media_type="application/octet-stream",
    )


@router.get("/library")
async def list_library(_: None = Depends(verify_api_key)):
    """List files in the library folder."""
    library_dir = get_docs_dir() / "library"
    if not library_dir.exists():
        return {"files": []}

    files = []
    for filepath in library_dir.glob("**/*"):
        if filepath.is_file():
            rel_path = filepath.relative_to(get_docs_dir())
            files.append({
                "name": filepath.name,
                "path": str(rel_path),
                "size": filepath.stat().st_size,
                "extension": filepath.suffix.lower(),
            })
    files.sort(key=lambda f: f["name"].lower())
    return {"files": files}


# Create temp app for MCP
temp_app = FastAPI()
temp_app.include_router(router)

# MCP routes: tag "mcp" becomes tools, rest excluded
route_maps = [
    RouteMap(tags={"mcp"}, mcp_type=MCPType.TOOL),
    RouteMap(mcp_type=MCPType.EXCLUDE),
]

mcp = FastMCP.from_fastapi(app=temp_app, route_maps=route_maps)
mcp_app = mcp.http_app(path="/", transport="streamable-http", stateless_http=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    WHORL_DIR.mkdir(parents=True, exist_ok=True)
    get_docs_dir().mkdir(parents=True, exist_ok=True)
    yield


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    async with lifespan(app):
        async with mcp_app.lifespan(app):
            yield


app = FastAPI(title="Whorl", description="Knowledge ingestion server", lifespan=combined_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")
app.mount("/mcp", mcp_app)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"


@app.get("/{full_path:path}")
async def serve_frontend(request: Request, full_path: str):
    """Serve frontend SPA - static files or index.html for client-side routing."""
    # Don't intercept API or MCP routes
    if full_path.startswith(("api/", "api", "mcp/", "mcp")):
        raise HTTPException(status_code=404, detail="Not found")
    # Try to serve static file first
    file_path = FRONTEND_DIR / full_path
    if full_path and file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    # Fall back to index.html for SPA routing
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not built. Run 'npm run build' in frontend/")
