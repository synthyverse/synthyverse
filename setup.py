from setuptools import setup, find_packages
import os
import sys

# Ensure scripts directory is in the path for imports
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.setup_utils import get_extras

version = "0.1.4"

# Define extras dynamically from requirements folder
extras = get_extras()

core_dependencies = []  # Always-installed dependencies

setup(
    name="synthyverse",
    version=version,
    description="Synthetic data generation and evaluation library",
    author="Jim Achterberg, Saif Ul Islam, Zia Ur Rehman",
    author_email=" ",
    packages=find_packages(),
    install_requires=core_dependencies,
    extras_require=extras,
    python_requires=">=3.8",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/synthyverse/synthyverse",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
