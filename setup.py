#! /usr/bin/env python

DESCRIPTION = "tubesML, a package that allows for flexible ML pipelines, model validation, and model inspection"
LONG_DESCRIPTION = """\
A package that allows for flexible ML pipelines, model validation, and model inspection. 
It builds on top of sklearn and preserves the data structure provided by pandas.
"""

DISTNAME = 'tubesml'
MAINTAINER = 'Luca Basanisi'
MAINTAINER_EMAIL = 'luca.basanisi@gmail.com'
URL = 'https://pypi.org/project/tubesml/'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/lucabasa/tubesML'
VERSION = '0.3.1'
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'numpy>=1.19.5',
    'pandas>=1.2.0',
    'scikit-learn>=0.24.1',
]


PACKAGES = [
    'tubesml',
]

CLASSIFIERS = [
    'Development Status :: 4 - Beta', 
    'Programming Language :: Python', 
    'Programming Language :: Python :: 3.8', 
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
]


if __name__ == "__main__":

    from setuptools import setup

    import sys
    if sys.version_info[:2] < (3, 6):
        raise RuntimeError("tubesml requires python >= 3.6.")

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
        classifiers=CLASSIFIERS
    )
