#!/usr/bin/env python3
"""Upload markdown files from a directory to whorl."""

import argparse
import asyncio
import fnmatch
import os
import sys
from pathlib import Path

import httpx


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract title from frontmatter if present."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = {}
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    frontmatter[key.strip()] = val.strip()
            return frontmatter, parts[2].strip()
    return {}, content


def matches_any(filepath: Path, patterns: list[str]) -> bool:
    """Check if filepath matches any of the glob patterns."""
    name = filepath.name
    path_str = str(filepath)
    for pattern in patterns:
        # Match filename, full path, or any path component
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(path_str, pattern):
            return True
        # Check if any parent directory matches (for excluding directories)
        for part in filepath.parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


async def upload_file(client: httpx.AsyncClient, filepath: Path, api_url: str, api_key: str, process: bool, context: str | None) -> bool:
    """Upload a single markdown file."""
    content = filepath.read_text()
    frontmatter, body = parse_frontmatter(content)

    # Use frontmatter title or filename
    title = frontmatter.pop("title", None) or filepath.stem

    # Pass remaining frontmatter as metadata (exclude id/created_at, server generates those)
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
            print(f"  = {filepath.name} (duplicate, skipped)")
        else:
            print(f"  + {filepath.name} -> {data['path']}")
        return True
    except httpx.HTTPStatusError as e:
        print(f"  x {filepath.name}: {e.response.status_code}")
        return False
    except Exception as e:
        print(f"  x {filepath.name}: {e}")
        return False


async def async_main(args, api_key: str, files: list[Path], batch_size: int):
    """Run uploads in parallel with batching."""
    sorted_files = sorted(files)
    total_success = 0

    async with httpx.AsyncClient() as client:
        for i in range(0, len(sorted_files), batch_size):
            batch = sorted_files[i:i + batch_size]
            tasks = [
                upload_file(client, f, args.url, api_key, args.process, args.context)
                for f in batch
            ]
            results = await asyncio.gather(*tasks)
            total_success += sum(results)

    return total_success


def main():
    parser = argparse.ArgumentParser(description="Upload markdown files to whorl")
    parser.add_argument("directory", nargs="?", type=Path, help="Directory containing .md files")
    parser.add_argument("--url", default="http://localhost:8000", help="Whorl API URL")
    parser.add_argument("--key", help="API key (or set WHORL_API_KEY env var)")
    parser.add_argument("--flat", "-f", action="store_true", help="Don't search recursively (default is recursive)")
    parser.add_argument("--process", "-p", action="store_true", help="Run ingestion agent on each file")
    parser.add_argument("--exclude", "-e", action="append", default=[], metavar="PATTERN",
                        help="Exclude files matching pattern (can be used multiple times)")
    parser.add_argument("--context", "-c", metavar="SOURCE",
                        help="Source context for all files (e.g., 'obsidian', 'meeting-notes')")
    parser.add_argument("--batch", "-b", type=int, default=25, metavar="N",
                        help="Number of files to upload in parallel (default: 25)")
    args = parser.parse_args()

    api_key = args.key or os.environ.get("WHORL_API_KEY")
    if not api_key:
        print("Error: API key required (--key or WHORL_API_KEY)")
        sys.exit(1)

    if not args.directory:
        parser.print_help()
        sys.exit(1)

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)

    pattern = "*.md" if args.flat else "**/*.md"
    files = [f for f in args.directory.glob(pattern) if not matches_any(f, args.exclude)]

    if not files:
        print(f"No .md files found in {args.directory}")
        sys.exit(0)

    print(f"Uploading {len(files)} file(s) from {args.directory}")
    if args.process:
        print("  (with agent processing)")

    success = asyncio.run(async_main(args, api_key, files, args.batch))

    print(f"\nDone: {success}/{len(files)} uploaded")


if __name__ == "__main__":
    main()
