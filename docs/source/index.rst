Welcome to synthyverse's documentation!
=========================================

**synthyverse** is an extensive ecosystem for synthetic data generation and evaluation in Python.

The synthyverse provides:

* 🔧 **Highly modular installation** - Install only those modules which you require to keep your installation lightweight.
* 📚 **Extensive library** - Any generator or metric can be quickly added without dependency conflicts due to synthyverse's modular installation. This allows the synthyverse to host a great amount of generators and evaluation metrics. It also allows the synthyverse to wrap around any existing synthetic data library.
* ⚙️ **Benchmarking module** - The benchmarking module executes a modular pipeline of synthetic data generation and evaluation. Choose a generator, set of evaluation metrics, and pipeline parameters, and obtain results on synthetic data quality.
* 👷 **Minimal preprocessing** - All preprocessing is handled by the synthyverse, so no need for scaling, one-hot encoding, or handling missing values. Different preprocessing schemes can be used by setting simple parameters.
* 👍 **Constraint support** - You can specify inter-column constraints which you want your synthetic data to follow. Constraints are modelled explicitly by the synthyverse, not through oversampling. This ensures efficient and reliable constraint setting.

Quick Start
-----------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started
   in_depth_usage

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index