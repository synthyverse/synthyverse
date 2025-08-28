from setuptools import setup, find_packages
import os

version = "0.1.1"


def read_requirements(filename):
    # Get the directory where setup.py is located
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(setup_dir, "requirements", filename)
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
    "tabddpm": base_requirements + read_requirements("generators/tabddpm.txt"),
    "tabsyn": base_requirements + read_requirements("generators/tabsyn.txt"),
    "cdtd": base_requirements + read_requirements("generators/cdtd.txt"),
    "tabargn": base_requirements + read_requirements("generators/tabargn.txt"),
    "realtabformer": base_requirements
    + read_requirements("generators/realtabformer.txt"),
    "ctabgan": base_requirements + read_requirements("generators/ctabgan.txt"),
}

# Create a "full" extra that includes all extras
extras["full"] = []
for key in extras:
    extras["full"].extend(extras[key])
extras["full"] = list(set(extras["full"]))

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
