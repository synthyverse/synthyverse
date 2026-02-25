#!/usr/bin/env python3
"""
Script to extract installation templates (extras) from setup.py and update
README.md and docs/source/getting_started.md with the templates list.
"""
import os
import re
import sys

# Add parent directory to path so we can import scripts.setup_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.setup_utils import get_extras_with_categories


def generate_templates_table(extras, categories):
    """Generate a markdown table of all available templates with categories."""
    # Map category names to display names and sort order
    category_display = {
        "generators": "Generator",
        "evaluation": "Evaluation",
        "all": "All",
    }

    category_order = {
        "generators": 0,
        "evaluation": 1,
        "all": 2,
    }

    # Sort extras: first by category order, then alphabetically within category
    # Put 'base' at the end of generators, and 'full' at the very end
    def sort_key(extra_name):
        if extra_name == "base":
            # Put base at the end of generators
            return (category_order.get("generators", 999), "zzzzzzz_base")
        if extra_name == "full":
            # Put full at the very end
            return (category_order.get("all", 999), "zzzzzzz_full")
        category = categories.get(extra_name, "unknown")
        return (category_order.get(category, 999), extra_name)

    sorted_extras = sorted(extras.keys(), key=sort_key)

    # Create table
    lines = [
        "## Available Installation Templates",
        "",
        "The following installation templates are available:",
        "",
        "| Template Name | Category | Installation Command |",
        "|---------------|----------|----------------------|",
    ]

    for extra in sorted_extras:
        category = categories.get(extra, "Unknown")
        category_label = category_display.get(category, category.title())
        lines.append(
            f"| `{extra}` | {category_label} | `pip install synthyverse[{extra}]` |"
        )

    lines.append("")
    lines.append(
        "**Note:** You can install multiple templates by separating them with commas, e.g., `pip install synthyverse[ctgan,eval]`"
    )

    return "\n".join(lines)


def update_markdown_file(file_path, templates_section):
    """Update a markdown file with the templates section."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the section after "See the overview of templates" line
    # We'll replace everything from that line until the next major section
    pattern = r"(See the \[overview of templates\][^\n]*\n)"

    # Check if templates section already exists
    if "## Available Installation Templates" in content:
        # Replace existing templates section
        pattern = r"(## Available Installation Templates.*?)(?=\n# |\n### |\Z)"
        new_content = re.sub(pattern, templates_section, content, flags=re.DOTALL)
    else:
        # Insert after the "See the overview of templates" line
        new_content = re.sub(
            pattern, r"\1\n" + templates_section + "\n", content, flags=re.DOTALL
        )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main():
    """Main function."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme_path = os.path.join(script_dir, "README.md")
    docs_path = os.path.join(script_dir, "docs", "source", "getting_started.md")

    if not os.path.exists(readme_path):
        print(f"Error: README.md not found at {readme_path}", file=sys.stderr)
        sys.exit(1)

    # Get extras with categories
    extras, categories = get_extras_with_categories()
    print(
        f"Found {len(extras)} installation templates: {', '.join(sorted(extras.keys()))}"
    )

    # Generate templates section
    templates_section = generate_templates_table(extras, categories)

    # Update README
    update_markdown_file(readme_path, templates_section)
    print(f"Successfully updated {readme_path}")

    # Update docs if it exists
    if os.path.exists(docs_path):
        update_markdown_file(docs_path, templates_section)
        print(f"Successfully updated {docs_path}")
    else:
        print(f"Warning: {docs_path} not found, skipping docs update")


if __name__ == "__main__":
    main()
