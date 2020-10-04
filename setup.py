#! /usr/bin/env python

DESCRIPTION = "pipeML, "
LONG_DESCRIPTION = """\
TBD
"""

DISTNAME = 'pipeml'
MAINTAINER = 'Luca Basanisi'
MAINTAINER_EMAIL = 'luca.basanisi@gmail.com'
URL = ''
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/lucabasa/pipeML'
VERSION = '0.0.1'
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'numpy=>1.19.2',
    'pandas>=1.1.2',
    'scikit-learn>=0.23.2',
]


PACKAGES = [
    'pipeml',
]

CLASSIFIERS = [
]


if __name__ == "__main__":

    from setuptools import setup

    import sys
    if sys.version_info[:2] < (3, 6):
        raise RuntimeError("pipeml requires python >= 3.6.")

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
