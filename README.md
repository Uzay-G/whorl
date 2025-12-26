# Whorl

Whorl is a system to scalably ingest all your personal documents (journals, writing, things you've read, etc...) and then make these accessible for you and for the AIs you interact with.

In doing so, you can effectively personalize your interactions with models, by making them query and get information from your whorl. Whorl gives AIs tools to fetch this knowldge, and also uses agents to trigger workflows whenever you add to your whorl - eg keeping lists of tasks, media recommendations, or ideas that might be worth exploring.

Whorl is designed to be simple, living in markdown files on your system, and then relying on agents to apply transformations, link these files, and enhance them in whatever way you'd like.

It is self-hosted, meaning that to have an AI use it, you need to run it from some public IP or endpoint, and then the mcp url is at `/mcp`.


## Installation

```bash
# Clone and install
{pip install, uv tool install} whorled
```

## Configuration

Create `~/.whorl/settings.json`:

```json
{
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
```

Set your password in `.env` or environment:

```bash
export WHORL_PASSWORD="your-secret-password"
export ANTHROPIC_API_KEY="your-anthropic-key"  # for AI features
```

## Running the Server

```bash
whorl server
```

The web UI will be available at `http://localhost:8000`.

## Ingestion

Now you will want to upload data to Whorl.

It is all stored in `~/.whorl/docs`, or `$WHORL_HOME_DIR/docs` in any kind of flat or foldered structure you want, so you can also just move files there.

If you have a folder with your notes, journals, etc... for this you can use the CLI, eg:

example command

Otherwise, we recommend using an agent, eg claude code, with instructions on the whorl server, or just moving files into the docs.

Add API format here for agent.


## Workflows

You can configure automated workflows that agents execute whenever you upload to whorl. These agents run bash commands on your files, so be careful with your data!

You can add my defaults using `whorl add-defaults`, which has:

- automatically summarizing and tagging new notes
- extracting references to todos and tasks into a task.md file
- gathering references to media into media.md
- collecting patterns of ideas into media.md

(implement this command, which also adds index.md)

Then, in `docs/index.md`, this has what renders on the whorl homepage. For me, it has references to the above files:

```
[[ideas.md]]

[[media.md]]

[[tasks.md]]
```

As you can see this is very flexible, go wild! And let me know what you find useful, I've just started experimenting with this.

## CLI Usage

document upload, sync, server,


arguments, be concise, clear, minimal

## API Endpoints

All endpoints require `X-Password` header.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ingest` | POST | Ingest new content |
| `/api/search` | POST | Full-text search |
| `/api/agent_search` | POST | AI-powered search |
| `/api/documents` | GET | List all documents |
| `/api/documents/{path}` | GET | Get document content |
| `/api/update` | POST | Update a document |
| `/api/delete` | POST | Delete a document |
| `/api/sync` | POST | Run missing agents on all docs |
| `/api/bash` | POST | Run bash commands in docs dir |
| `/api/health` | GET | Health check (no auth) |

## MCP Integration

Whorl exposes an MCP server at `/mcp` for integration with Claude Code and other MCP-compatible tools.

Add to your Claude MCP configuration (e.g. `~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "whorl": {
      "url": "http://localhost:8000/mcp",
      "type": "http",
      "headers": {
        "X-Password": "your-whorl-password"
      }
    }
  }
}
```

Available MCP tools:
- `ingest` - Add content to knowledge base
- `search` - Full-text search
- `agent_search` - AI-powered semantic search
- `bash` - Run commands in the docs directory

## Ingestion Agents

Create markdown prompts in `~/.whorl/prompts/ingestion/` to process documents during ingestion. Each prompt file becomes an agent that runs on ingested content.

Example prompt (`~/.whorl/prompts/ingestion/summarize.md`):

```markdown
---
model: sonnet
max_turns: 10
---
You are a document summarizer. Given the document at {filepath}, create a summary file.

Use the bash and str_replace_editor tools to read the document and create a summary.
```

## Building the Frontend

```bash
cd frontend
npm install
npm run build
```

## Project Structure

```
whorl/
├── whorl/
│   ├── server.py      # FastAPI server with MCP integration
│   ├── models.py      # Pydantic request/response models
│   ├── agents.py      # Agent loop and ingestion/search agents
│   ├── cli.py         # CLI tool
│   └── lib/
│       ├── utils.py       # Hash index, frontmatter, path validation
│       └── text_edit.py   # Text editor tool for agents
├── frontend/          # Vue.js web UI
└── pyproject.toml
```

## License

MIT
