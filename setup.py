import versioneer
from setuptools import find_packages, setup

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
)
