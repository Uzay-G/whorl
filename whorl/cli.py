#!/usr/bin/env python3
"""Whorl CLI - upload and manage documents."""

import argparse
import asyncio
import fnmatch
import json
import os
import sys
from pathlib import Path

import httpx
import yaml

WHORL_DIR = Path.home() / ".whorl"
SETTINGS_PATH = WHORL_DIR / "settings.json"


def load_settings() -> dict:
    """Load settings from ~/.whorl/settings.json."""
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    return {}


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract frontmatter from markdown content."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
                return frontmatter, parts[2].strip()
            except yaml.YAMLError:
                pass
    return {}, content


def matches_any(filepath: Path, patterns: list[str]) -> bool:
    """Check if filepath matches any of the glob patterns."""
    name = filepath.name
    path_str = str(filepath)
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(path_str, pattern):
            return True
        for part in filepath.parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


async def upload_file(client: httpx.AsyncClient, filepath: Path, api_url: str, api_key: str, process: bool, context: str | None) -> bool:
    """Upload a single markdown file."""
    content = filepath.read_text()
    frontmatter, body = parse_frontmatter(content)

    title = frontmatter.pop("title", None) or filepath.stem
    frontmatter.pop("id", None)
    frontmatter.pop("created_at", None)
    if context:
        frontmatter["source"] = context
    metadata = frontmatter if frontmatter else None

    try:
        resp = await client.post(
            f"{api_url}/ingest",
            json={"content": body, "title": title, "metadata": metadata, "process": process},
            headers={"X-API-Key": api_key},
            timeout=600 if process else 30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("duplicate"):
            print(f"  = {filepath.name} (duplicate)")
        else:
            print(f"  + {filepath.name} -> {data['path']}")
        return True
    except httpx.HTTPStatusError as e:
        print(f"  x {filepath.name}: {e.response.status_code}")
        return False
    except Exception as e:
        print(f"  x {filepath.name}: {e}")
        return False


async def reingest_file(client: httpx.AsyncClient, doc_path: str, api_url: str, api_key: str) -> tuple[bool, bool]:
    """Trigger re-ingestion on a document. Returns (success, skipped)."""
    try:
        resp = await client.post(
            f"{api_url}/reingest",
            json={"path": doc_path},
            headers={"X-API-Key": api_key},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        skipped = data.get("skipped", False)
        if skipped:
            print(f"  = {doc_path} (complete)")
        else:
            print(f"  + {doc_path} ({len(data.get('processed', []))} agents)")
        return True, skipped
    except httpx.HTTPStatusError as e:
        print(f"  x {doc_path}: {e.response.status_code}")
        return False, False
    except Exception as e:
        print(f"  x {doc_path}: {e}")
        return False, False


async def run_upload(args, api_key: str):
    """Run upload command."""
    pattern = "*.md" if args.flat else "**/*.md"
    files = [f for f in args.directory.glob(pattern) if not matches_any(f, args.exclude)]

    if not files:
        print(f"No .md files found in {args.directory}")
        return

    print(f"Uploading {len(files)} file(s) from {args.directory}")
    if args.process:
        print("  (with agent processing)")

    sorted_files = sorted(files)
    total_success = 0

    async with httpx.AsyncClient() as client:
        for i in range(0, len(sorted_files), args.batch):
            batch = sorted_files[i:i + args.batch]
            tasks = [upload_file(client, f, args.url, api_key, args.process, args.context) for f in batch]
            results = await asyncio.gather(*tasks)
            total_success += sum(results)

    print(f"\nDone: {total_success}/{len(files)} uploaded")


async def run_reingest(args, api_key: str):
    """Run reingest command."""
    async with httpx.AsyncClient() as client:
        # Get all documents
        resp = await client.get(f"{args.url}/documents", headers={"X-API-Key": api_key})
        resp.raise_for_status()
        docs = resp.json()["docs"]

        if not docs:
            print("No documents found")
            return

        print(f"Checking {len(docs)} document(s) for missing agents...")

        total_processed = 0
        total_skipped = 0
        for i in range(0, len(docs), args.batch):
            batch = docs[i:i + args.batch]
            tasks = [reingest_file(client, doc["path"], args.url, api_key) for doc in batch]
            results = await asyncio.gather(*tasks)
            for success, skipped in results:
                if success:
                    if skipped:
                        total_skipped += 1
                    else:
                        total_processed += 1

        print(f"\nDone: {total_processed} processed, {total_skipped} already complete")


def cmd_upload(args):
    """Upload command handler."""
    api_key = args.key or os.environ.get("WHORL_API_KEY")
    if not api_key:
        print("Error: API key required (--key or WHORL_API_KEY)")
        sys.exit(1)

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)

    asyncio.run(run_upload(args, api_key))


def cmd_reingest(args):
    """Reingest command handler."""
    api_key = args.key or os.environ.get("WHORL_API_KEY")
    if not api_key:
        print("Error: API key required (--key or WHORL_API_KEY)")
        sys.exit(1)

    asyncio.run(run_reingest(args, api_key))


def main():
    settings = load_settings()
    default_url = settings.get("api_base", "http://localhost:8000")

    parser = argparse.ArgumentParser(description="Whorl CLI")
    parser.add_argument("--url", default=default_url, help=f"Whorl API URL (default: {default_url})")
    parser.add_argument("--key", help="API key (or set WHORL_API_KEY env var)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload markdown files")
    upload_parser.add_argument("directory", type=Path, help="Directory containing .md files")
    upload_parser.add_argument("--flat", "-f", action="store_true", help="Don't search recursively")
    upload_parser.add_argument("--process", "-p", action="store_true", help="Run ingestion agents")
    upload_parser.add_argument("--exclude", "-e", action="append", default=[], metavar="PATTERN",
                               help="Exclude files matching pattern")
    upload_parser.add_argument("--context", "-c", metavar="SOURCE", help="Source context for files")
    upload_parser.add_argument("--batch", "-b", type=int, default=50, help="Batch size (default: 50)")
    upload_parser.set_defaults(func=cmd_upload)

    # Reingest command
    reingest_parser = subparsers.add_parser("reingest", help="Re-run ingestion on docs missing agents")
    reingest_parser.add_argument("--batch", "-b", type=int, default=5, help="Batch size (default: 5)")
    reingest_parser.set_defaults(func=cmd_reingest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
