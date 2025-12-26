# Whorl

Whorl is a system to scalably ingest all your personal documents (journals, writing, things you've read, etc...) and then make these accessible for you and for the AIs you interact with.

In doing so, you can effectively personalize your interactions with models, by making them query and get information from your whorl. Whorl gives AIs tools to fetch this knowldge, and also uses agents to trigger workflows whenever you add to your whorl - eg keeping lists of tasks, media recommendations, or ideas that might be worth exploring.

Whorl is designed to be simple, living in markdown files on your system, and then relying on agents to apply transformations, link these files, and enhance them in whatever way you'd like.

It is self-hosted, meaning that to have an AI use it, you need to run it from some public IP or endpoint, and then the mcp url is at `/mcp`.


## Getting started


Install:

```bash
pip install whorled
# or
uv tool install whorled
```


Setup default configuration:

```
whorl init

```

If you want AI features, set your anthropic API key in `.env` or environment:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"  # for AI features
```

## Running the Server

```bash
whorl server
```

The web UI will be available at `http://localhost:8000`.

## Ingestion

Now you will want to upload data to Whorl.

It is all stored in `~/.whorl/docs` (or `$WHORL_HOME/docs`) in any kind of flat or foldered structure you want, so you can also just move files there.

If you have a folder with your notes, journals, etc... you can use the CLI:

```bash
whorl upload ~/notes --context "personal-notes"
```

Otherwise, use an agent with the whorl MCP server, or just move files into the docs directory.

For agents, the ingest API is:

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -H "X-Password: $WHORL_PASSWORD" \
  -d '{"content": "Your note content here", "title": "Note Title", "metadata": {}}'
```


## Workflows

You can configure automated workflows that agents execute whenever you upload to whorl. These agents run bash commands on your files, so be careful with your data!

You can add some defaults using `whorl init`, which sets up:

- automatically summarizing and tagging new notes
- extracting todos and tasks into `tasks.md`
- gathering media references into `media.md`
- collecting ideas into `ideas.md`

Then, in `docs/index.md`, you can reference these files (renders on the whorl homepage):

```
[[ideas.md]]
[[media.md]]
[[tasks.md]]
```

This is very flexible - go wild! Let me know what you find useful, I've just started experimenting with this.

## CLI Usage

```bash
whorl server                     # start server on :8000
whorl server --port 3000         # custom port
whorl server --reload            # dev mode with auto-reload

whorl upload <dir>               # upload text files from directory
whorl upload <dir> -f            # flat (non-recursive)
whorl upload <dir> -e "*.draft"  # exclude pattern
whorl upload <dir> -c "source"   # add source context

whorl sync                       # run missing agents on all docs
whorl init                       # set up default prompts and index
```

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
- `text_search` - Full-text search
- `agent_search` - AI search
- `bash` - Run commands in the docs directory

You can add something to your global system prompt with the MCP, and then tell your chat assistant to use these tools when it might benefit from context about you.

## Custom Ingestion Agents

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


Feel free to contribute! Note because this was a quick side project I was pretty lax on supervision with claude code, which was very useful.

## License

MIT
