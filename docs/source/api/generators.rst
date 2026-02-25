Generators
==========

All generators inherit from a base generator class (currently :class:`~synthyverse.generators.base.TabularBaseGenerator` for tabular data), which provides powerful preprocessing capabilities including constraints and missing value handling.

For detailed information on using these Base Generator parameters (constraints, missing_imputation_method, retain_missingness), see the :doc:`../in_depth_usage` guide.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   generators/arf
   generators/bn
   generators/ctabgan
   generators/ctgan
   generators/cdtd
   generators/forestdiffusion
   generators/nrgboost
   generators/permutation
   generators/realtabformer
   generators/smote
   generators/synthpop
   generators/tabargn
   generators/tabddpm
   generators/tabsyn
   generators/tvae
   generators/unmaskingtrees
