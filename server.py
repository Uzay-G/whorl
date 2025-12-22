import json
import os
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime
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


class IngestRequest(BaseModel):
    content: str
    metadata: dict[str, Any] | None = None
    title: str | None = None
    process: bool = False


class IngestResponse(BaseModel):
    id: str
    path: str


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


async def run_ingestion_agent(content: str, filepath: Path, settings: dict) -> None:
    config = settings["ingestion_config"]
    prompt_path = WHORL_DIR / config["prompt"]
    prompt_template = prompt_path.read_text()
    system_prompt = prompt_template.format(filepath=filepath)

    options = ClaudeAgentOptions(
        allowed_tools=config["allowed_tools"],
        max_turns=config["max_turns"],
        system_prompt=system_prompt,
        model=config.get("model", "sonnet"),
        cwd=str(WHORL_DIR),
    )

    async for _ in query(prompt=content, options=options):
        pass


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

    doc_id = os.urandom(4).hex()
    timestamp = datetime.utcnow().isoformat()

    filename = f"{doc_id}.md"
    if request.title:
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in request.title)
        safe_title = safe_title.strip().replace(" ", "-")[:50]
        if safe_title:
            filename = f"{doc_id}-{safe_title}.md"

    filepath = docs_dir / filename

    frontmatter = {
        "id": doc_id,
        "created_at": timestamp,
    }
    if request.title:
        frontmatter["title"] = request.title
    if request.metadata:
        frontmatter.update(request.metadata)

    yaml_frontmatter = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    doc_content = f"---\n{yaml_frontmatter}---\n\n{request.content}"
    filepath.write_text(doc_content)

    if request.process:
        await run_ingestion_agent(request.content, filepath, settings)

    return IngestResponse(id=doc_id, path=str(filepath))


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
