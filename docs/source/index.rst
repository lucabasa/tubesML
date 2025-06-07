Welcome to tubesML's documentation!
===================================

A package that allows for flexible ML pipelines, model validation, and model inspection.

The project started to productize code written for Kaggle competitions and it is mostly designed to facilitate the creation of flexible processing and modeling pipeline. Everything is built around scikit-learn transformers and methods, but it is possible to extend the concept to other ML packages.

The main characteristic that makes the pipeline components of tubesML flexible is the fact that it always preserves the pandas DataFrame structure, hence making it easy to, for example, create a feature within a pipeline and test it in a grid search.

The package contains also several methods to inspect the models and its predictions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :titlesonly:

   intro
   base
   clean
   process
   feat_en
   utility
   model_selection
   model_comparison
   data_exploration
   report
   models
   error_analysis


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
