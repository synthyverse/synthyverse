from __future__ import annotations
import os, sys
from datetime import date

# --- Make package importable during the build ---
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

project = "synthyverse"
author = "Jim Achterberg, Saif Ul Islam, Zia Ur Rehman"
copyright = f"{date.today().year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
]

# Theme
html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["api/generated/**"]

# Logo - path relative to conf.py
html_logo = "_static/logo.png"

# MyST Markdown
myst_enable_extensions = ["deflist", "colon_fence", "substitution", "linkify"]

# Autosummary / Autodoc
# Curated API docs only: do not generate autosummary stub pages.
autosummary_generate = False
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
# Put type hints into the docs text, not the function signature
autodoc_typehints = "description"
typehints_fully_qualified = False

# Napoleon: prefer Google style
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # <- keep False to avoid mixed parsing
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_include_init_with_doc = False
napoleon_attr_annotations = True

# skips these imports during doc build (TBD: automate reading all package imports)
autodoc_mock_imports = [
    "pandas",
    "numpy",
    "scikit-learn",
    "sklearn",
    "torch",
    "pytorch",
    "pytorch_lightning",
    "torch_ema",
    "scipy",
    "tqdm",
    "matplotlib",
    "seaborn",
    "torchtuples",
    "copulas",
    "geomloss",
    "joblib",
    "sdv",
    "sdmetrics",
    "shap",
    "tsai",
    "xgboost",
    "optuna",
    "xgbse",
    "synthcity",
    "tensorflow",
    "nrgboost",
    "realtabformer",
    "ctgan",
    "utrees",
    "ForestDiffusion",
    "python-synthpop",
    "synthpop",
    "arfpy",
    "mostlyai",
    "einops",
    "lightgbm",
    "icecream",
    "catboost",
    "pydantic",
    "category_encoders",
    "zero",
    "tomli-w",
    "tomli_w",
    "tomli",
    "jax",
    "six",
    "scripts",
    "imbalanced-learn",
    "imblearn",
]

# Cross-refs to other package docs: potentially to be added later
