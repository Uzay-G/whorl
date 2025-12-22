import asyncio
import hashlib
import json
import os
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from claude_agent_sdk import query, ClaudeAgentOptions
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType

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


def verify_api_key(x_api_key: str):
    if not WHORL_API_KEY:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if x_api_key != WHORL_API_KEY:
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


def search_docs(query_str: str, limit: int = 10, context: int = 2) -> list[SearchResult]:
    """Full text search using ripgrep."""
    docs_dir = get_docs_dir()
    if not docs_dir.exists():
        return []

    try:
        result = subprocess.run(
            [
                "rg",
                "--json",
                "-i",  # case insensitive
                "-C", str(context),  # context lines
                query_str,
                str(docs_dir),
            ],
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


async def run_single_ingestion_agent(content: str, filepath: Path, prompt_path: Path, config: dict) -> str | None:
    """Run a single ingestion agent with a specific prompt. Returns the prompt name or None on failure."""
    try:
        prompt_template = prompt_path.read_text()
        system_prompt = prompt_template.format(filepath=filepath)

        options = ClaudeAgentOptions(
            allowed_tools=config["allowed_tools"],
            max_turns=config["max_turns"],
            system_prompt=system_prompt,
            model=config.get("model", "sonnet"),
            cwd=str(WHORL_DIR),
            permission_mode="acceptEdits",
        )

        async for message in query(prompt=content, options=options):
            # Log agent messages for debugging
            if hasattr(message, 'type'):
                print(f"[{prompt_path.stem}] {message.type}: {getattr(message, 'content', '')[:100] if hasattr(message, 'content') else ''}")

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
    prompt_template = prompt_path.read_text()
    docs_dir = get_docs_dir()
    system_prompt = prompt_template.format(docs_dir=docs_dir)

    options = ClaudeAgentOptions(
        allowed_tools=config["allowed_tools"],
        max_turns=config["max_turns"],
        system_prompt=system_prompt,
        model=config.get("model", "sonnet"),
        cwd=str(docs_dir),
        permission_mode="acceptEdits",
    )

    result_text = ""
    async for message in query(prompt=query_str, options=options):
        if hasattr(message, "result") and message.result:
            result_text = message.result

    return result_text or "No results found."


@router.post("/ingest", response_model=IngestResponse, tags=["mcp"])
async def ingest(request: IngestRequest, x_api_key: str = Header(...)):
    """Ingest content into the knowledge base."""
    verify_api_key(x_api_key)

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
async def search(request: SearchRequest, x_api_key: str = Header(...)):
    """Full text search over the knowledge base."""
    verify_api_key(x_api_key)

    results = search_docs(request.query, request.limit, request.context)
    return SearchResponse(results=results, total=len(results))


@router.post("/agent_search", response_model=AgentSearchResponse, tags=["mcp"])
async def agent_search(request: AgentSearchRequest, x_api_key: str = Header(...)):
    """Agent-powered search over the knowledge base."""
    verify_api_key(x_api_key)

    settings = load_settings()
    answer = await run_search_agent(request.query, settings)
    return AgentSearchResponse(answer=answer)


@router.post("/bash", response_model=BashResponse, tags=["mcp"])
async def bash(request: BashRequest, x_api_key: str = Header(...)):
    """Run a bash command in the docs directory."""
    verify_api_key(x_api_key)

    docs_dir = get_docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            request.command,
            shell=True,
            cwd=str(docs_dir),
            capture_output=True,
            text=True,
            timeout=request.timeout,
        )
        return BashResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Command timed out")


@router.post("/delete")
async def delete_doc(request: DeleteRequest, x_api_key: str = Header(...)):
    """Delete a document."""
    verify_api_key(x_api_key)

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
async def update_doc(request: UpdateRequest, x_api_key: str = Header(...)):
    """Update an existing document."""
    verify_api_key(x_api_key)

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


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/documents")
async def list_docs(x_api_key: str = Header(...)):
    """List all documents."""
    verify_api_key(x_api_key)
    docs_dir = get_docs_dir()
    docs = []
    for filepath in docs_dir.glob("*.md"):
        content = filepath.read_text()
        frontmatter, body = parse_frontmatter(content)
        docs.append({
            "id": frontmatter.get("id", filepath.stem),
            "path": filepath.name,
            "title": frontmatter.get("title"),
            "created_at": frontmatter.get("created_at"),
            "frontmatter": frontmatter,
        })
    docs.sort(key=lambda d: d.get("created_at") or "", reverse=True)
    return {"docs": docs}


@router.get("/documents/{path:path}")
async def get_doc(path: str, x_api_key: str = Header(...)):
    """Get a document's content."""
    verify_api_key(x_api_key)
    docs_dir = get_docs_dir()
    filepath = docs_dir / path

    # Security check
    if not str(filepath.resolve()).startswith(str(docs_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Not found")

    return {"content": filepath.read_text()}


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
app.include_router(router)
app.mount("/mcp", mcp_app)
