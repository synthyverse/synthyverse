#!/usr/bin/env python3
"""
Script to extract installation templates (extras) from setup.py and update README.md
"""
import os
import re
import sys

# Add parent directory to path so we can import scripts.setup_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.setup_utils import get_extras


def generate_templates_table(extras):
    """Generate a markdown table of all available templates."""
    # Sort extras alphabetically, but put 'base' and 'full' at the end
    sorted_extras = sorted([k for k in extras.keys() if k not in ["base", "full"]])
    if "base" in extras:
        sorted_extras.append("base")
    if "full" in extras:
        sorted_extras.append("full")

    # Create table
    lines = [
        "## Available Installation Templates",
        "",
        "The following installation templates are available:",
        "",
        "| Template Name | Installation Command |",
        "|---------------|----------------------|",
    ]

    for extra in sorted_extras:
        lines.append(f"| `{extra}` | `pip install synthyverse[{extra}]` |")

    lines.append("")
    lines.append(
        "**Note:** You can install multiple templates by separating them with commas, e.g., `pip install synthyverse[ctgan,eval]`"
    )

    return "\n".join(lines)


def update_readme(readme_path, templates_section):
    """Update README.md with the templates section."""
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the section after "See the overview of templates" line
    # We'll replace everything from that line until the next major section
    pattern = r"(See the \[overview of templates\][^\n]*\n)"

    # Check if templates section already exists
    if "## Available Installation Templates" in content:
        # Replace existing templates section
        pattern = r"(## Available Installation Templates.*?)(?=\n# |\Z)"
        new_content = re.sub(pattern, templates_section, content, flags=re.DOTALL)
    else:
        # Insert after the "See the overview of templates" line
        new_content = re.sub(
            pattern, r"\1\n" + templates_section + "\n", content, flags=re.DOTALL
        )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main():
    """Main function."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme_path = os.path.join(script_dir, "README.md")

    if not os.path.exists(readme_path):
        print(f"Error: README.md not found at {readme_path}", file=sys.stderr)
        sys.exit(1)

    # Get extras
    extras = get_extras()
    print(
        f"Found {len(extras)} installation templates: {', '.join(sorted(extras.keys()))}"
    )

    # Generate templates section
    templates_section = generate_templates_table(extras)

    # Update README
    update_readme(readme_path, templates_section)
    print(f"Successfully updated {readme_path}")


if __name__ == "__main__":
    main()
