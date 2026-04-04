#! /usr/bin/env python

DESCRIPTION = "tubesML, a package that allows for flexible ML pipelines, model validation, and model inspection"
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()

DISTNAME = "tubesml"
MAINTAINER = "Luca Basanisi"
MAINTAINER_EMAIL = "luca.basanisi@gmail.com"
URL = "https://pypi.org/project/tubesml/"
LICENSE = "Apache 2.0"
DOWNLOAD_URL = "https://github.com/lucabasa/tubesML"
PROJECT_URLS = {
    "Documentation": "https://tubesml.readthedocs.io/",
    "Source Code": "https://github.com/lucabasa/tubesML",
}

VERSION = "1.1.0"
PYTHON_REQUIRES = ">=3.10"
INSTALL_REQUIRES = [
    "kneed==0.8.5",
    "matplotlib>=3.10.0",
    "pandas>=2.3.3",
    "scikit-learn>=1.2.2",
    "scipy>=1.16.3",
    "seaborn>=0.13.2",
    "shap>=0.50.0",
]

PACKAGES = [
    "tubesml",
]

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.10",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
]

TEST_REQUIRE = ["pytest==9.0.2", "coverage==7.13.5", "xgboost>=3.2.0", "lightgbm>=4.6.0"]


if __name__ == "__main__":
    from setuptools import setup

    import sys

    if sys.version_info[:2] < (3, 11):
        raise RuntimeError("tubesml requires python >= 3.11.")

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
        classifiers=CLASSIFIERS,
    )
