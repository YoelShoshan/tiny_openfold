#!/usr/bin/env python
import os
import pathlib
from setuptools import setup, find_namespace_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# list of requirements
requirements = []
with open(os.path.join(HERE, "requirements.txt"), "r") as fh:
    for line in fh:
        if not line.startswith("#"):
            requirements.append(line.strip())


setup(
    name="tiny_openfold",
    version="0.0.1",
    description="low dependencies subset of OpenFold helper functions",
    url="https://github.ibm.com/YOELS/tiny_openfold",
    author="",
    author_email="",
    packages=find_namespace_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    license="Apache License 2.0", 
    install_requires=requirements,
    
)
