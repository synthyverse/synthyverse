Generators
==========

Low-level generators inherit from :class:`~synthyverse.generators.base.BaseGenerator` and expose a common ``fit`` / ``generate`` interface. Shared preprocessing, constraints, and schema restoration are handled by :class:`~synthyverse.generators.base.DataProcessor` or the higher-level :class:`~synthyverse.generators.base.SynthyverseGenerator` wrapper.

For detailed examples of low-level generators, preprocessing, persistence, and wrappers, see the :doc:`../in_depth_usage` guide.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   generators/base
   generators/arf
   generators/ctgan
   generators/cdtd
   generators/smote
   generators/tabddpm
   generators/tabdiff
   generators/tabsyn
   generators/tvae
   generators/univariate
   generators/synthpop
