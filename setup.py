from setuptools import setup

# these lines allow the version to be specified in Makefile.private
import os
version = os.environ.get("MODULEVER", "0.0")

setup(
    name = 'fit_lib', version = version,
    packages = ['fit_lib'],
    include_package_data = True, zip_safe = False)
