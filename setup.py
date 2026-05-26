#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="ectil",
    version="0.0.1",
    # The project targets Python 3.10; an exact "==3.10.9" pin was too strict and
    # broke editable installs when conda resolved a newer 3.10.x patch (e.g. 3.10.20),
    # which newer pip/setuptools now enforce on `python_requires`.
    python_requires=">=3.10,<3.11",
    description="ECTIL: Label-efficient Computational Tumour Infiltrating Lymphocyte (TIL) assessment in breast cancer",
    author="Yoni Schirris",
    author_email="yschirris@gmail.com",
    url="https://github.com/YoniSchirris/ectil",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
