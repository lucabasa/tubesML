.. |PythonMinVersion| replace:: 3.6
.. |NumPyMinVersion| replace:: 1.19.5
.. |PandasMinVersion| replace:: 1.2.0
.. |SklearnMinVersion| replace:: 0.24.1


TubesML
=======

A package that allows for flexible ML pipelines, model validation, and model inspection.

The project started to productize code written for Kaggle competitions and it is mostly designed to facilitate the creation of flexible processing and modeling pipeline. Everything is built around scikit-learn transformers and methods, but it is possible to extend the concept to other ML packages.

The main characteristic that makes the pipeline components of **tubesML** flexible is the fact that it always preserves the pandas DataFrame structure, hence making it easy to, for example, create a feature within a pipeline and test it in a grid search.


Installation
------------

Dependencies
~~~~~~~~~~~~
tubesML requires:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- Pandas (>= |PandasMinVersion|)
- Scikit-Learn (>= |SklearnMinVersion|)

=======

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of pandas and sklearn,
the easiest way to install scikit-learn is using ``pip``   ::

    pip install -U tubesml
    
Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/lucabasa/tubesML
- Download releases: https://pypi.org/project/tubesml/
- Documentation: https://tubesml.readthedocs.io/en/latest/
    