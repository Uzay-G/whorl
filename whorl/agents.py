"""Agent loop and ingestion/search agent functions."""

import asyncio
import traceback
from pathlib import Path

import anthropic

from whorl.lib import text_edit
from whorl.lib.utils import parse_frontmatter, write_doc_with_frontmatter


ANTHROPIC_TOOLS = [
    {"type": "text_editor_20250124", "name": "str_replace_editor"},
    {"type": "bash_20250124", "name": "bash"},
]

MODEL_MAP = {
    "haiku": "claude-sonnet-4-20250514",  # Use sonnet as haiku replacement for tool use
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
}


def run_bash_command(command: str, cwd: str, timeout: int = 30) -> tuple[str, str, int]:
    """Execute a bash command and return (stdout, stderr, returncode)."""
    import subprocess
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", -1
    except Exception as e:
        return "", str(e), -1


def format_bash_output(stdout: str, stderr: str, returncode: int) -> str:
    """Format bash output for agent consumption."""
    output = stdout
    if stderr:
        output += f"\n[stderr]: {stderr}" if output else f"[stderr]: {stderr}"
    if returncode != 0:
        output += f"\n[exit code: {returncode}]"
    return output or "(no output)"


def truncate_if_huge(output: str, limit: int = 100000) -> str:
    """Only truncate if output is very large (100k+ chars)."""
    if len(output) <= limit:
        return output
    return output[:limit] + f"\n\n[truncated - was {len(output)} chars]"


def execute_tool(name: str, inputs: dict, cwd: str) -> str:
    """Execute a tool and return the result (truncated if huge)."""
    if name == "bash":
        command = inputs.get("command", "")
        print(f"  [bash] {command[:80]}{'...' if len(command) > 80 else ''}")
        stdout, stderr, returncode = run_bash_command(command, cwd)
        return truncate_if_huge(format_bash_output(stdout, stderr, returncode))

    elif name == "str_replace_editor":
        command = inputs.get("command", "")
        path = inputs.get("path", "")
        # Make path absolute relative to cwd
        if path and not path.startswith("/"):
            path = str(Path(cwd) / path)
        # Security: ensure path stays within cwd
        try:
            resolved = Path(path).resolve()
            cwd_resolved = Path(cwd).resolve()
            if not str(resolved).startswith(str(cwd_resolved)):
                return f"Error: path {path} is outside allowed directory"
        except Exception:
            return f"Error: invalid path {path}"
        print(f"  [edit] {command} {path}")
        # Remove command/path from inputs since we're passing them explicitly
        kwargs = {k: v for k, v in inputs.items() if k not in ("command", "path")}
        return truncate_if_huge(text_edit.execute(command, path, **kwargs))

    return f"Unknown tool: {name}"


async def run_agent_loop(
    client: anthropic.AsyncAnthropic,
    system_prompt: str,
    user_message: str,
    model: str,
    max_turns: int,
    cwd: str,
) -> str:
    """Run a multi-turn agent conversation with bash and text_editor tools."""
    model_id = MODEL_MAP.get(model, model)
    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = await client.beta.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system_prompt,
            tools=ANTHROPIC_TOOLS,
            messages=messages,
            betas=["computer-use-2025-01-24"],
        )

        # Check if we're done (no tool use)
        has_tool_use = any(block.type == "tool_use" for block in response.content)

        if not has_tool_use:
            # Extract final text response
            text_parts = [block.text for block in response.content if block.type == "text"]
            return "\n".join(text_parts) if text_parts else ""

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input, cwd)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn":
            break

    return ""


async def run_single_ingestion_agent(content: str, filepath: Path, prompt_path: Path, config: dict, cwd: str) -> str | None:
    """Run a single ingestion agent with a specific prompt. Returns the prompt name or None on failure."""
    try:
        prompt_raw = prompt_path.read_text()
        prompt_frontmatter, prompt_body = parse_frontmatter(prompt_raw)
        system_prompt = prompt_body.format(filepath=filepath)

        # Config from frontmatter, fall back to global config
        model = prompt_frontmatter.get("model", config.get("model", "sonnet"))
        max_turns = prompt_frontmatter.get("max_turns", config.get("max_turns", 50))

        # Truncate very large content to avoid token limits (~150k chars = ~40k tokens)
        max_content_chars = 150000
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + f"\n\n[truncated - was {len(content)} chars]"

        client = anthropic.AsyncAnthropic()

        await run_agent_loop(
            client=client,
            system_prompt=system_prompt,
            user_message=content,
            model=model,
            max_turns=max_turns,
            cwd=cwd,
        )

        return prompt_path.stem
    except Exception as e:
        print(f"Agent {prompt_path.stem} failed: {e}")
        traceback.print_exc()
        return None


async def process_doc_with_agents(
    filepath: Path,
    body: str,
    prompt_files: list[Path],
    config: dict,
    cwd: str,
) -> list[str]:
    """Run multiple agents on a document and return list of successful agent names."""
    tasks = [
        run_single_ingestion_agent(body, filepath, prompt_path, config, cwd)
        for prompt_path in prompt_files if prompt_path.exists()
    ]
    results = await asyncio.gather(*tasks)
    return [name for name in results if name is not None]


DEFAULT_SEARCH_PROMPT = """You are a search assistant for a personal knowledge base at {docs_dir}.

Use bash to search files (rg, cat, ls) and answer the user's question.
Synthesize information from multiple documents if needed.
Cite sources when referencing specific documents.
"""


async def run_search_agent(query_str: str, prompt_path: Path | None, config: dict, cwd: str) -> str:
    """Run the search agent to answer a query."""
    if prompt_path and prompt_path.exists():
        prompt_raw = prompt_path.read_text()
        prompt_frontmatter, prompt_body = parse_frontmatter(prompt_raw)
    else:
        prompt_frontmatter = {}
        prompt_body = DEFAULT_SEARCH_PROMPT

    system_prompt = prompt_body.format(docs_dir=cwd)

    # Config from frontmatter, fall back to global config
    model = prompt_frontmatter.get("model", config.get("model", "sonnet"))
    max_turns = prompt_frontmatter.get("max_turns", config.get("max_turns", 25))

    client = anthropic.AsyncAnthropic()

    result = await run_agent_loop(
        client=client,
        system_prompt=system_prompt,
        user_message=query_str,
        model=model,
        max_turns=max_turns,
        cwd=cwd,
    )

    return result or "No results found."
