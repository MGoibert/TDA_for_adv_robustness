#!/usr/bin/env python3

import setuptools
import os


def _read_reqs(relpath):
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines()
                if (s.strip() and not s.startswith("#"))]


_REQUIREMENTS_TXT = _read_reqs("requirements.txt")
_INSTALL_REQUIRES = [l for l in _REQUIREMENTS_TXT if "://" not in l]

setuptools.setup(
    name='tda',
    version='0.2',
    install_requires=_INSTALL_REQUIRES,
    data_files=[('.', ['requirements.txt'])],
    packages=setuptools.find_packages(),
)
