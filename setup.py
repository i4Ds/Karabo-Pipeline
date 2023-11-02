import versioneer
from setuptools import find_packages, setup

setup(
    version=versioneer.get_version(),
    packages=find_packages(),
)
