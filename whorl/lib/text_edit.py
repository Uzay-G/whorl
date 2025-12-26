"""Text editor tool implementation for Anthropic's text_editor tool."""

from pathlib import Path


def view(path: str, view_range: list[int] | None = None) -> str:
    """View file contents, optionally a specific line range."""
    p = Path(path)
    if not p.exists():
        return f"Error: {path} does not exist"

    lines = p.read_text().splitlines(keepends=True)

    if view_range:
        start, end = view_range[0] - 1, view_range[1]  # 1-indexed to 0-indexed
        start = max(0, start)
        end = min(len(lines), end)
        lines = lines[start:end]
        line_offset = start
    else:
        line_offset = 0

    # Format with line numbers
    result = []
    for i, line in enumerate(lines):
        result.append(f"{i + 1 + line_offset:4d}\t{line.rstrip()}")

    return "\n".join(result) if result else "(empty file)"


def create(path: str, file_text: str) -> str:
    """Create a new file with the given content."""
    p = Path(path)
    if p.exists():
        return f"Error: {path} already exists"

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(file_text)
    return f"Created {path}"


def str_replace(path: str, old_str: str, new_str: str) -> str:
    """Replace old_str with new_str in the file. old_str must be unique."""
    p = Path(path)
    if not p.exists():
        return f"Error: {path} does not exist"

    content = p.read_text()
    count = content.count(old_str)

    if count == 0:
        return f"Error: old_str not found in {path}"
    if count > 1:
        return f"Error: old_str appears {count} times in {path}, must be unique"

    new_content = content.replace(old_str, new_str, 1)
    p.write_text(new_content)
    return f"Replaced in {path}"


def insert(path: str, insert_line: int, new_str: str) -> str:
    """Insert new_str after the given line number (1-indexed). Use 0 to insert at beginning."""
    p = Path(path)
    if not p.exists():
        return f"Error: {path} does not exist"

    lines = p.read_text().splitlines(keepends=True)

    if insert_line < 0 or insert_line > len(lines):
        return f"Error: insert_line {insert_line} out of range (0-{len(lines)})"

    # Ensure new_str ends with newline
    if new_str and not new_str.endswith("\n"):
        new_str += "\n"

    lines.insert(insert_line, new_str)
    p.write_text("".join(lines))
    return f"Inserted at line {insert_line} in {path}"


def execute(command: str, path: str, **kwargs) -> str:
    """Execute a text editor command."""
    handlers = {
        "view": view,
        "create": create,
        "str_replace": str_replace,
        "insert": insert,
    }

    if command not in handlers:
        return f"Error: unknown command {command}"

    try:
        if command == "view":
            return handlers[command](path, kwargs.get("view_range"))
        elif command == "create":
            return handlers[command](path, kwargs.get("file_text", ""))
        elif command == "str_replace":
            return handlers[command](path, kwargs.get("old_str", ""), kwargs.get("new_str", ""))
        elif command == "insert":
            return handlers[command](path, kwargs.get("insert_line", 0), kwargs.get("new_str", ""))
    except Exception as e:
        return f"Error: {e}"
