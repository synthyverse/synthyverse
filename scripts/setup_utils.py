"""
Shared utilities for setup.py and related scripts.
Contains functions for reading requirements and extracting extras.
"""

import os


def get_setup_dir():
    """Get the directory where setup.py is located (project root)."""
    # This file is in scripts/, so go up one level to get project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_requirements(filename):
    """
    Read requirements from a file in the requirements directory.

    Args:
        filename: Relative path from requirements directory (e.g., "generators/base.txt")

    Returns:
        List of requirement strings
    """
    setup_dir = get_setup_dir()
    filepath = os.path.join(setup_dir, "requirements", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def get_extras():
    """
    Extract extras from requirements directory.

    Returns:
        Dictionary mapping extra names to lists of requirements
    """
    extras, _ = get_extras_with_categories()
    return extras


def get_extras_with_categories():
    """
    Extract extras from requirements directory with category information.

    Returns:
        Tuple of (extras_dict, category_dict) where:
        - extras_dict: Dictionary mapping extra names to lists of requirements
        - category_dict: Dictionary mapping extra names to their category (generators/evaluation/imputers)
    """
    setup_dir = get_setup_dir()
    requirements_dir = os.path.join(setup_dir, "requirements")

    extras = {}
    categories = {}

    # Process each subdirectory in requirements folder
    for subdir in ["generators", "evaluation", "imputers"]:
        subdir_path = os.path.join(requirements_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Check if there's a base.txt file in this subdirectory
        base_file = os.path.join(subdir_path, "base.txt")
        base_requirements = []
        if os.path.exists(base_file):
            base_requirements = read_requirements(f"{subdir}/base.txt")
            # Add base extra only for generators (to maintain backward compatibility)
            if subdir == "generators":
                extras["base"] = base_requirements
                categories["base"] = "generators"

        # Process all .txt files in the subdirectory
        for filename in os.listdir(subdir_path):
            if filename.endswith(".txt"):
                name = filename[:-4]  # Remove .txt extension

                # Skip base.txt as it's handled separately
                if filename == "base.txt":
                    continue

                # Read requirements for this file
                file_requirements = read_requirements(f"{subdir}/{filename}")

                # Combine with base requirements if they exist
                extras[name] = base_requirements + file_requirements
                categories[name] = subdir

    # Create a "full" extra that includes all extras
    extras["full"] = []
    for key in extras:
        extras["full"].extend(extras[key])
    extras["full"] = list(set(extras["full"]))
    categories["full"] = "all"

    return extras, categories
