Feature Engineering
===================

In this section, we find classes to create new features. This includes

- creating polynomial features
- target encode categorical features
- using PCA to create new features or compress the data

In every case, the output is going to be a pandas DataFrame so that any further manipulation of the data can be as easy as the first.

.. automodule:: tubesml.poly
    :members:

.. automodule:: tubesml.encoders
    :members:

.. automodule:: tubesml.pca
    :members:
