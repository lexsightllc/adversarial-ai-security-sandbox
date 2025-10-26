#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Pre-commit hook to insert SPDX headers for supported file types."""

from __future__ import annotations

import sys
from pathlib import Path

LICENSE_IDENTIFIER = "SPDX-License-Identifier: MPL-2.0"

HEADER_STYLES = {
    ".py": ("# ", "", True),
    ".pyi": ("# ", "", True),
    ".js": ("// ", "", True),
    ".jsx": ("// ", "", True),
    ".ts": ("// ", "", True),
    ".tsx": ("// ", "", True),
    ".css": ("/* ", " */", True),
    ".scss": ("/* ", " */", True),
    ".yml": ("# ", "", True),
    ".yaml": ("# ", "", True),
    ".toml": ("# ", "", True),
    ".sh": ("# ", "", True),
    ".txt": ("# ", "", True),
    ".cfg": ("# ", "", True),
    ".ini": ("# ", "", True),
}

DOCKERFILE_NAMES = {"dockerfile", "dockerfile.dev", "dockerfile.prod"}


def determine_style(path: Path) -> tuple[str, str, bool] | None:
    suffix = path.suffix.lower()
    if suffix in HEADER_STYLES:
        return HEADER_STYLES[suffix]

    name = path.name.lower()
    if name in DOCKERFILE_NAMES or name.startswith("dockerfile"):
        return "# ", "", True

    return None


def add_header(path: Path) -> None:
    style = determine_style(path)
    if style is None:
        return

    prefix, suffix, blank_line_after = style

    try:
        original_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return

    if LICENSE_IDENTIFIER in original_text.splitlines()[0:5]:
        return

    lines = original_text.splitlines()
    header_line = f"{prefix}{LICENSE_IDENTIFIER}{suffix}".rstrip()

    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1

    new_lines = lines.copy()
    new_lines.insert(insert_at, header_line)
    if blank_line_after:
        new_lines.insert(insert_at + 1, "")

    path.write_text("\n".join(new_lines) + ("\n" if original_text.endswith("\n") else ""), encoding="utf-8")


def main(argv: list[str]) -> int:
    for filename in argv[1:]:
        add_header(Path(filename))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
