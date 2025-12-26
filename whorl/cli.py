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

WHORL_DIR = Path(os.environ.get("WHORL_HOME", Path.home() / ".whorl"))
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


async def upload_file(client: httpx.AsyncClient, filepath: Path, api_url: str, password: str, process: bool, context: str | None) -> bool:
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
            f"{api_url}/api/ingest",
            json={"content": body, "title": title, "metadata": metadata, "process": process},
            headers={"X-Password": password},
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


async def run_sync(api_url: str, password: str):
    """Call the sync endpoint to process all documents with missing agents."""
    print("Syncing documents...")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{api_url}/api/sync",
                headers={"X-Password": password},
                timeout=1800,  # 30 min timeout for large collections
            )
            resp.raise_for_status()
            data = resp.json()

            for result in data.get("results", []):
                if result["status"] == "skipped":
                    print(f"  = {result['path']}")
                else:
                    print(f"  + {result['path']} ({len(result['agents'])} agents)")

            print(f"\nDone: {data['processed']} processed, {data['skipped']} skipped")
        except httpx.HTTPStatusError as e:
            print(f"Error: {e.response.status_code}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


TEXT_EXTENSIONS = {".md", ".txt", ".text", ".markdown"}


async def run_upload(args, password: str):
    """Run upload command."""
    pattern = "*" if args.flat else "**/*"
    files = [
        f for f in args.directory.glob(pattern)
        if f.is_file() and f.suffix.lower() in TEXT_EXTENSIONS and not matches_any(f, args.exclude)
    ]

    if not files:
        print(f"No text files found in {args.directory}")
        return

    print(f"Uploading {len(files)} file(s) from {args.directory}")
    if args.process:
        print("  (with agent processing)")

    sorted_files = sorted(files)
    total_success = 0

    async with httpx.AsyncClient() as client:
        for i in range(0, len(sorted_files), args.batch):
            batch = sorted_files[i:i + args.batch]
            tasks = [upload_file(client, f, args.url, password, args.process, args.context) for f in batch]
            results = await asyncio.gather(*tasks)
            total_success += sum(results)

    print(f"\nDone: {total_success}/{len(files)} uploaded")


def cmd_upload(args):
    """Upload command handler."""
    password = args.password or os.environ.get("WHORL_PASSWORD")
    if not password:
        print("Error: Password required (--password or WHORL_PASSWORD)")
        sys.exit(1)

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)

    asyncio.run(run_upload(args, password))


def cmd_sync(args):
    """Sync command handler."""
    password = args.password or os.environ.get("WHORL_PASSWORD")
    if not password:
        print("Error: Password required (--password or WHORL_PASSWORD)")
        sys.exit(1)

    asyncio.run(run_sync(args.url, password))


def cmd_server(args):
    """Server command handler."""
    import uvicorn
    uvicorn.run("whorl.server:app", host=args.host, port=args.port, reload=args.reload)


DEFAULT_SETTINGS = {
    "docs_dir": "docs",
    "api_base": "http://localhost:8000",
    "ingestion_config": {
        "prompts_dir": "prompts/ingestion",
        "model": "sonnet",
        "max_turns": 50
    },
    "search_config": {
        "prompt": "prompts/search.md",
        "model": "sonnet",
        "max_turns": 25,
        "exclude": []
    }
}

DEFAULT_PROMPTS = {
    "summarize": '''---
model: sonnet
max_turns: 10
---
You are processing a document at {filepath}.

Read the file, then update its YAML frontmatter to add:
- summary: a 1-2 sentence summary
- tags: relevant tags as a list

Use bash to read and str_replace_editor to update the file.
''',
    "tasks": '''---
model: sonnet
max_turns: 15
---
You are processing a document at {filepath}.

Read the file and extract any todos, tasks, or action items.
Append them to tasks.md in the same directory (create if needed).
Format: "- [ ] task description (from: filename)"
''',
    "media": '''---
model: sonnet
max_turns: 15
---
You are processing a document at {filepath}.

Read the file and extract any media recommendations (books, movies, articles, music, etc).
Append them to media.md in the same directory (create if needed).
Format: "- **type**: title - context/notes"
''',
    "ideas": '''---
model: sonnet
max_turns: 15
---
You are processing a document at {filepath}.

Read the file and extract any interesting ideas, insights, or things worth exploring.
Append them to ideas.md in the same directory (create if needed).
Format: "- idea description (from: filename)"
'''
}

DEFAULT_INDEX = '''# Whorl

Welcome to your personal knowledge base.

## Quick Links

[[tasks.md]]
[[media.md]]
[[ideas.md]]
'''

DEFAULT_SEARCH_PROMPT = '''---
model: sonnet
max_turns: 25
---
You are a search assistant for a personal knowledge base at {docs_dir}.

Use bash to search files (rg, cat, ls) and answer the user's question.
Synthesize information from multiple documents if needed.
'''


def cmd_init(args):
    """Initialize whorl with default configuration."""
    import getpass

    print("Whorl Setup")
    print("=" * 40)

    # Check if already initialized
    if SETTINGS_PATH.exists():
        resp = input(f"\n{SETTINGS_PATH} already exists. Overwrite? [y/N] ").strip().lower()
        if resp != 'y':
            print("Keeping existing settings.")
        else:
            _write_settings()
    else:
        _write_settings()

    # Password
    print("\n" + "-" * 40)
    password = getpass.getpass("Set password (leave empty to skip): ")
    if password:
        with open(SETTINGS_PATH) as f:
            settings = json.load(f)
        settings["password"] = password
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
        print("Password saved to settings.json")

    # Prompts
    print("\n" + "-" * 40)
    prompts_dir = WHORL_DIR / "prompts" / "ingestion"
    resp = input(f"Set up default ingestion prompts in {prompts_dir}? [Y/n] ").strip().lower()
    if resp != 'n':
        prompts_dir.mkdir(parents=True, exist_ok=True)
        for name, content in DEFAULT_PROMPTS.items():
            prompt_file = prompts_dir / f"{name}.md"
            if prompt_file.exists():
                print(f"  Skipping {name}.md (exists)")
            else:
                prompt_file.write_text(content)
                print(f"  Created {name}.md")

        # Search prompt
        search_prompt = WHORL_DIR / "prompts" / "search.md"
        if not search_prompt.exists():
            search_prompt.parent.mkdir(parents=True, exist_ok=True)
            search_prompt.write_text(DEFAULT_SEARCH_PROMPT)
            print(f"  Created search.md")

    # Index
    print("\n" + "-" * 40)
    docs_dir = WHORL_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    index_file = docs_dir / "index.md"
    resp = input(f"Create {index_file}? [Y/n] ").strip().lower()
    if resp != 'n':
        if index_file.exists():
            print("  Skipping (exists)")
        else:
            index_file.write_text(DEFAULT_INDEX)
            print("  Created index.md")

    print("\n" + "=" * 40)
    print("Done! Run 'whorl server' to start.")


def _write_settings():
    """Write default settings."""
    WHORL_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(DEFAULT_SETTINGS, f, indent=2)
    print(f"Created {SETTINGS_PATH}")


def main():
    settings = load_settings()
    default_url = settings.get("api_base", "http://localhost:8000")

    parser = argparse.ArgumentParser(description="Whorl CLI")
    parser.add_argument("--url", default=default_url, help=f"Whorl API URL (default: {default_url})")
    parser.add_argument("--password", help="Password (or set WHORL_PASSWORD env var)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload markdown files")
    upload_parser.add_argument("directory", type=Path, help="Directory containing text files (.md, .txt)")
    upload_parser.add_argument("--flat", "-f", action="store_true", help="Don't search recursively")
    upload_parser.add_argument("--process", "-p", action="store_true", help="Run ingestion agents")
    upload_parser.add_argument("--exclude", "-e", action="append", default=[], metavar="PATTERN",
                               help="Exclude files matching pattern")
    upload_parser.add_argument("--context", "-c", metavar="SOURCE", help="Source context for files")
    upload_parser.add_argument("--batch", "-b", type=int, default=50, help="Batch size (default: 50)")
    upload_parser.set_defaults(func=cmd_upload)

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Run missing ingestion agents on all documents")
    sync_parser.set_defaults(func=cmd_sync)

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the whorl server")
    server_parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    server_parser.set_defaults(func=cmd_server)

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize whorl with default configuration")
    init_parser.set_defaults(func=cmd_init)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
