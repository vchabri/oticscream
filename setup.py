"""
Setup script for oticscream
==========================

This script allows to install oticscream within the python environment.

Usage
-----
::

    python setup.py install

"""
import re
import os
from distutils.core import setup

from setuptools import find_packages

# Get the version from __init__.py
path = os.path.join(os.path.dirname(__file__), 'oticscream', '__init__.py')
with open(path) as f:
    version_file = f.read()

version = re.search(r"^\s*__version__\s*=\s*['\"]([^'\"]+)['\"]",
                    version_file, re.M)
if version:
    version = version.group(1)
else:
    raise RuntimeError("Unable to find version string.")

# Long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(

    # library name
    name='oticscream',

    # code version
    version=version,

    # list libraries to be imported
    packages=find_packages(),

    # Descriptions
    description="oticscream",
    long_description=long_description,

    # List of dependancies
    setup_requires=['pytest-runner'],
    install_requires=['numpy>=1.13',
                      'openturns'],
    tests_require=['pytest'],

    # Enable to take into account MANIFEST.in
    # include_package_data=True,
    license="LGPL"
)
