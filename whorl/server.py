"""Whorl - Personal knowledge ingestion server."""

import hashlib
import json
import logging
import os
import subprocess
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType

from whorl.models import (
    IngestRequest, IngestResponse,
    SearchRequest, SearchResult, SearchResponse,
    AgentSearchRequest, AgentSearchResponse,
    BashRequest, BashResponse,
    DeleteRequest, UpdateRequest,
    SyncResult, SyncResponse,
)
from whorl.agents import process_doc_with_agents, run_search_agent, run_bash_command
from whorl.lib.utils import (
    HashIndex,
    parse_frontmatter,
    write_doc_with_frontmatter,
    validate_path_in_dir,
)

load_dotenv()

# Constants
WHORL_DIR = Path(os.environ.get("WHORL_HOME", Path.home() / ".whorl"))
SETTINGS_PATH = WHORL_DIR / "settings.json"
HASH_INDEX_PATH = WHORL_DIR / "hash-index.json"
FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"

# Global state
router = APIRouter()
hash_index = HashIndex(HASH_INDEX_PATH)
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


def get_password() -> str | None:
    """Get password from env var or settings."""
    return os.environ.get("WHORL_PASSWORD") or load_settings().get("password")


# Auth
password_header = APIKeyHeader(name="X-Password", auto_error=False)


async def verify_password(password: str | None = Depends(password_header)):
    """Verify password from header."""
    expected = get_password()
    if not expected:
        raise HTTPException(status_code=500, detail="Server password not configured")
    if not password or password != expected:
        raise HTTPException(status_code=401, detail="Invalid password")


# Search helper
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
            msg_type = data.get("type")
            if msg_type not in ("match", "context"):
                continue

            d = data.get("data", {})
            path = d.get("path", {}).get("text")
            lines_data = d.get("lines", {})
            line_text = lines_data.get("text", "").strip() if isinstance(lines_data, dict) else ""
            line_num = d.get("line_number", 0)

            if not path or not line_text:
                continue

            if path not in matches:
                matches[path] = {"lines": [], "match_count": 0}
            matches[path]["lines"].append((line_num, line_text))
            if msg_type == "match":
                matches[path]["match_count"] += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    # Build results
    results = []
    for filepath_str, match_data in matches.items():
        filepath = Path(filepath_str)
        try:
            content = filepath.read_text(errors="replace")
            frontmatter, _ = parse_frontmatter(content)
        except Exception:
            frontmatter = {}

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


# Routes
@router.post("/ingest", response_model=IngestResponse, tags=["mcp"])
async def ingest(request: IngestRequest, _: None = Depends(verify_password)):
    """Ingest content into the knowledge base."""
    docs_dir = get_docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings()

    # Compute content hash and check for duplicates
    content_hash = compute_content_hash(request.content)
    existing = hash_index.find(content_hash)

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

    write_doc_with_frontmatter(filepath, frontmatter, request.content)
    hash_index.add(content_hash, doc_id, filepath.name)

    if request.process:
        config = settings.get("ingestion_config", {})
        prompts_dir = WHORL_DIR / config.get("prompts_dir", "prompts/ingestion")
        if prompts_dir.exists():
            prompt_files = list(prompts_dir.glob("*.md"))
            if prompt_files:
                agent_names = await process_doc_with_agents(
                    filepath, request.content, prompt_files, config, str(docs_dir)
                )
                # Update frontmatter with processed agents
                file_content = filepath.read_text()
                frontmatter, body = parse_frontmatter(file_content)
                frontmatter["processed"] = sorted(agent_names)
                write_doc_with_frontmatter(filepath, frontmatter, body)

    return IngestResponse(id=doc_id, path=filepath.name, content_hash=content_hash)


@router.post("/text_search", response_model=SearchResponse, tags=["mcp"])
async def text_search(request: SearchRequest, _: None = Depends(verify_password)):
    """Full text search over the knowledge base."""
    settings = load_settings()
    exclude = settings.get("search_config", {}).get("exclude", [])
    results = search_docs(request.query, request.limit, request.context, exclude)
    return SearchResponse(results=results, total=len(results))


@router.post("/agent_search", response_model=AgentSearchResponse, tags=["mcp"])
async def agent_search(request: AgentSearchRequest, _: None = Depends(verify_password)):
    """Agent-powered search over the knowledge base."""
    settings = load_settings()
    config = settings.get("search_config", {})
    prompt_path = WHORL_DIR / config.get("prompt", "prompts/search.md")
    docs_dir = get_docs_dir()
    answer = await run_search_agent(request.query, prompt_path, config, str(docs_dir))
    return AgentSearchResponse(answer=answer)


@router.post("/bash", response_model=BashResponse, tags=["mcp"])
async def bash(request: BashRequest, _: None = Depends(verify_password)):
    """Run a bash command in the docs directory."""
    docs_dir = get_docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)

    stdout, stderr, returncode = run_bash_command(request.command, str(docs_dir), request.timeout)
    if returncode == -1 and stderr == "Command timed out":
        raise HTTPException(status_code=408, detail="Command timed out")
    return BashResponse(stdout=stdout, stderr=stderr, returncode=returncode)


@router.post("/delete")
async def delete_doc(request: DeleteRequest, _: None = Depends(verify_password)):
    """Delete a document."""
    docs_dir = get_docs_dir()
    filepath = docs_dir / request.path

    try:
        filepath = validate_path_in_dir(filepath, docs_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from hash index
    content = filepath.read_text()
    frontmatter, _ = parse_frontmatter(content)
    if content_hash := frontmatter.get("content_hash"):
        hash_index.remove(content_hash)

    filepath.unlink()
    return {"status": "deleted", "path": request.path}


@router.post("/update")
async def update_doc(request: UpdateRequest, _: None = Depends(verify_password)):
    """Update an existing document."""
    docs_dir = get_docs_dir()
    filepath = docs_dir / request.path

    try:
        filepath = validate_path_in_dir(filepath, docs_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    # Read existing frontmatter
    existing_content = filepath.read_text()
    frontmatter, _ = parse_frontmatter(existing_content)

    # Update title if provided
    if request.title is not None:
        frontmatter["title"] = request.title

    write_doc_with_frontmatter(filepath, frontmatter, request.content)
    return {"status": "updated", "path": request.path}


@router.post("/sync", response_model=SyncResponse)
async def sync_docs(_: None = Depends(verify_password)):
    """Sync all documents - run missing ingestion agents on each."""
    docs_dir = get_docs_dir()
    settings = load_settings()
    config = settings.get("ingestion_config", {})
    prompts_dir = WHORL_DIR / config.get("prompts_dir", "prompts/ingestion")

    # Get expected agents
    expected_agents = {p.stem for p in prompts_dir.glob("*.md")} if prompts_dir.exists() else set()

    if not expected_agents:
        return SyncResponse(total=0, processed=0, skipped=0, results=[])

    # Find all markdown docs
    doc_files = list(docs_dir.glob("**/*.md"))

    if not doc_files:
        return SyncResponse(total=0, processed=0, skipped=0, results=[])

    # Process docs sequentially to avoid overwhelming the system
    results = []
    for filepath in doc_files:
        rel_path = str(filepath.relative_to(docs_dir))

        try:
            content = filepath.read_text()
            frontmatter, body = parse_frontmatter(content)
        except Exception:
            results.append(SyncResult(path=rel_path, status="skipped", agents=[]))
            continue

        processed_agents = set(frontmatter.get("processed", []))
        missing_agents = expected_agents - processed_agents

        if not missing_agents:
            results.append(SyncResult(path=rel_path, status="skipped", agents=list(processed_agents)))
            continue

        # Run only missing agents
        prompt_files = [prompts_dir / f"{agent}.md" for agent in missing_agents]
        new_agents = await process_doc_with_agents(filepath, body, prompt_files, config, str(docs_dir))

        # Update frontmatter with all processed agents
        all_processed = sorted(processed_agents | set(new_agents))
        file_content = filepath.read_text()
        frontmatter, body = parse_frontmatter(file_content)
        frontmatter["processed"] = all_processed
        write_doc_with_frontmatter(filepath, frontmatter, body)

        results.append(SyncResult(path=rel_path, status="processed", agents=all_processed))

    processed = sum(1 for r in results if r.status == "processed")
    skipped = sum(1 for r in results if r.status == "skipped")

    return SyncResponse(total=len(results), processed=processed, skipped=skipped, results=results)


@router.get("/settings")
async def get_settings(_: None = Depends(verify_password)):
    """Get current settings."""
    return load_settings()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/documents")
async def list_docs(_: None = Depends(verify_password)):
    """List all documents (recursive)."""
    docs_dir = get_docs_dir()
    if not docs_dir.exists():
        return {"docs": []}
    docs = []
    for filepath in docs_dir.glob("**/*.md"):
        try:
            content = filepath.read_text(errors="replace")
            frontmatter, body = parse_frontmatter(content)
        except Exception:
            frontmatter = {}
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
async def get_doc(path: str, _: None = Depends(verify_password)):
    """Get a document's content."""
    docs_dir = get_docs_dir()
    filepath = docs_dir / path

    try:
        validate_path_in_dir(filepath, docs_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Not found")

    return {"content": filepath.read_text()}


@router.get("/download/{path:path}")
async def download_file(
    path: str,
    password: str | None = None,
    header_password: str | None = Depends(password_header),
):
    """Download a file from the docs directory (PDFs, etc)."""
    # Accept password from either query param or header (for browser downloads)
    pwd = password or header_password
    expected = get_password()
    if not expected:
        raise HTTPException(status_code=500, detail="Server password not configured")
    if not pwd or pwd != expected:
        raise HTTPException(status_code=401, detail="Invalid password")

    docs_dir = get_docs_dir()
    filepath = docs_dir / path

    try:
        validate_path_in_dir(filepath, docs_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(
        filepath,
        filename=filepath.name,
        media_type="application/octet-stream",
    )


@router.get("/library")
async def list_library(_: None = Depends(verify_password)):
    """List files in the library folder."""
    library_dir = get_docs_dir() / "library"
    if not library_dir.exists():
        return {"files": []}

    files = []
    for filepath in library_dir.glob("**/*"):
        if filepath.is_file() and not filepath.name.startswith("."):
            rel_path = filepath.relative_to(get_docs_dir())
            files.append({
                "name": filepath.name,
                "path": str(rel_path),
                "size": filepath.stat().st_size,
                "extension": filepath.suffix.lower(),
            })
    files.sort(key=lambda f: f["name"].lower())
    return {"files": files}


# MCP setup
temp_app = FastAPI()
temp_app.include_router(router)

route_maps = [
    RouteMap(tags={"mcp"}, mcp_type=MCPType.TOOL),
    RouteMap(mcp_type=MCPType.EXCLUDE),
]

mcp = FastMCP.from_fastapi(app=temp_app, route_maps=route_maps)
mcp_app = mcp.http_app(path="/", transport="streamable-http", stateless_http=True)


# App setup
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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log all unhandled exceptions."""
    print(f"ERROR: {request.method} {request.url.path}")
    traceback.print_exc()
    raise exc


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
    raise HTTPException(status_code=404, detail="Frontend not built")
