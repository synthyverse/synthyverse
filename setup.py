from setuptools import setup, find_packages
import os
from synthyverse.__version__ import __version__


# Utility: read requirements from file
def read_requirements(filename):
    filepath = os.path.join("requirements", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


# Define extras dynamically from requirements folder
base_requirements = read_requirements("generators/base.txt")
extras = {
    "eval": read_requirements("evaluation/eval.txt"),
    "ctgan": base_requirements + read_requirements("generators/ctgan.txt"),
    "arf": base_requirements + read_requirements("generators/arf.txt"),
    "bn": base_requirements + read_requirements("generators/bn.txt"),
    "tvae": base_requirements + read_requirements("generators/tvae.txt"),
}

# Create a "full" extra that includes all extras
extras["full"] = []
for key in extras:
    extras["full"].extend(extras[key])
# Optionally remove duplicates
extras["full"] = list(set(extras["full"]))

core_dependencies = []  # Always-installed dependencies

setup(
    name="synthyverse",
    version=__version__,
    description="Synthetic data generation and evaluation library",
    author="Jim Achterberg, Saif Ul Islam, Zia Ur Rehman",
    author_email=" ",
    packages=find_packages(),
    install_requires=core_dependencies,
    extras_require=extras,
    python_requires=">=3.8",
    long_description=open("README.md").read(),
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
