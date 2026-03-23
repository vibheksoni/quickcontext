import re
from pathlib import Path
from typing import Optional


PROJECT_MARKERS = [
    ".git",
    "package.json",
    "Cargo.toml",
    "pyproject.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "composer.json",
    "Gemfile",
    ".project",
]


def find_project_root(start_path: str | Path) -> Optional[Path]:
    """
    Find project root by walking up directory tree looking for project markers.

    Args:
        start_path: Starting directory or file path

    Returns:
        Path to project root, or None if no project root found
    """
    current = Path(start_path).resolve()

    if current.is_file():
        current = current.parent

    start = current
    home = Path.home().resolve()

    while current != current.parent:
        if current == home and current != start:
            break
        for marker in PROJECT_MARKERS:
            if (current / marker).exists():
                return current

        current = current.parent

    return None


def sanitize_project_name(name: str) -> str:
    """
    Sanitize project name to be valid Qdrant collection name.

    Qdrant collection names must:
    - Start with a letter or underscore
    - Contain only letters, digits, underscores, hyphens
    - Be between 1-255 characters

    Args:
        name: Raw project name (e.g., directory name)

    Returns:
        Sanitized collection name
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    if not sanitized:
        sanitized = "project"

    if not sanitized[0].isalpha() and sanitized[0] != "_":
        sanitized = f"_{sanitized}"

    sanitized = sanitized[:255]

    return sanitized


def detect_project_name(directory: str | Path, manual_override: Optional[str] = None) -> str:
    """
    Detect project name from directory path.

    Args:
        directory: Directory to index
        manual_override: Manual project name override

    Returns:
        Sanitized project name for use as collection name
    """
    if manual_override:
        return sanitize_project_name(manual_override)

    project_root = find_project_root(directory)

    if project_root:
        return sanitize_project_name(project_root.name)

    directory_path = Path(directory).resolve()
    return sanitize_project_name(directory_path.name)
