"""Utility functions for whorl."""

import json
from pathlib import Path

import yaml


def validate_path_in_dir(path: Path, base_dir: Path) -> Path:
    """Validate that path is within base_dir. Returns resolved path or raises ValueError."""
    try:
        resolved = path.resolve()
        base_resolved = base_dir.resolve()
        if not str(resolved).startswith(str(base_resolved)):
            raise ValueError(f"Path {path} is outside allowed directory")
        return resolved
    except Exception as e:
        raise ValueError(f"Invalid path: {path}") from e


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


def write_doc_with_frontmatter(filepath: Path, frontmatter: dict, body: str) -> None:
    """Write a document with YAML frontmatter."""
    yaml_frontmatter = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    filepath.write_text(f"---\n{yaml_frontmatter}---\n\n{body}")


class HashIndex:
    """Manages content hash index for deduplication."""

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self._cache: dict[str, dict] | None = None

    def _load(self) -> dict[str, dict]:
        if self._cache is None:
            if self.index_path.exists():
                with open(self.index_path) as f:
                    self._cache = json.load(f)
            else:
                self._cache = {}
        return self._cache

    def _save(self) -> None:
        if self._cache is not None:
            with open(self.index_path, "w") as f:
                json.dump(self._cache, f, indent=2)

    def find(self, content_hash: str) -> dict | None:
        """Find an existing doc by content hash. O(1) lookup."""
        return self._load().get(content_hash)

    def add(self, content_hash: str, doc_id: str, path: str) -> None:
        """Add a document to the hash index."""
        index = self._load()
        index[content_hash] = {"id": doc_id, "path": path}
        self._save()

    def remove(self, content_hash: str) -> None:
        """Remove a document from the hash index."""
        index = self._load()
        if content_hash in index:
            del index[content_hash]
            self._save()
