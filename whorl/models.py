"""Pydantic models for whorl API."""

from typing import Any

from pydantic import BaseModel


# Ingest
class IngestRequest(BaseModel):
    content: str
    metadata: dict[str, Any] | None = None
    title: str | None = None
    process: bool = True


class IngestResponse(BaseModel):
    id: str
    path: str
    duplicate: bool = False
    content_hash: str


# Search
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    context: int = 2


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


# Bash
class BashRequest(BaseModel):
    command: str
    timeout: int = 30


class BashResponse(BaseModel):
    stdout: str
    stderr: str
    returncode: int


# Document operations
class DeleteRequest(BaseModel):
    path: str


class UpdateRequest(BaseModel):
    path: str
    content: str
    title: str | None = None


# Sync
class SyncResult(BaseModel):
    path: str
    status: str  # "processed" or "skipped"
    agents: list[str]


class SyncResponse(BaseModel):
    total: int
    processed: int
    skipped: int
    results: list[SyncResult]
