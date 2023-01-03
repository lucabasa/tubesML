#! /usr/bin/env python

DESCRIPTION = "tubesML, a package that allows for flexible ML pipelines, model validation, and model inspection"
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

DISTNAME = 'tubesml'
MAINTAINER = 'Luca Basanisi'
MAINTAINER_EMAIL = 'luca.basanisi@gmail.com'
URL = 'https://pypi.org/project/tubesml/'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/lucabasa/tubesML'
PROJECT_URLS = {
    #'Bug Tracker': ,
    'Documentation': 'https://tubesml.readthedocs.io/',
    'Source Code': 'https://github.com/lucabasa/tubesML'
}
VERSION = '0.6.1'
PYTHON_REQUIRES = ">=3.7"

INSTALL_REQUIRES = [
    'matplotlib>=3.5.3',
    'pandas>=1.3.5',
    'scikit-learn>=1.0.2',
    'seaborn>=0.12.0'
]


PACKAGES = [
    'tubesml',
]

CLASSIFIERS = [
    'Development Status :: 4 - Beta', 
    'Programming Language :: Python', 
    'Programming Language :: Python :: 3.10', 
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
]

TEST_REQUIRE = [
        "pytest==7.1.3", 
        "coverage==6.4.4", 
        'xgboost==1.6.2',
        'lightgbm==3.3.2'
]



if __name__ == "__main__":

    from setuptools import setup

    import sys
    if sys.version_info[:2] < (3, 7):
        raise RuntimeError("tubesml requires python >= 3.7.")

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
        test_require=TEST_REQUIRE,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
        classifiers=CLASSIFIERS
    )
