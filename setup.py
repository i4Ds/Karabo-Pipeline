import os
import re

from setuptools import setup

with open(os.path.join("karabo", "version.py"), mode="r") as file:
    version_txt = file.readline()

canonical_pattern = r"([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?"  # noqa: E501
karabo_version = re.search(canonical_pattern, version_txt).group()

# implicitly takes config from setup.cfg
setup(
    version=karabo_version,
)
