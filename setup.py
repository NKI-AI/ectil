#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="ectil",
    version="0.0.1",
    python_requires="==3.10.9",
    description="ECTIL: Label-efficient Computational Tumour Infiltrating Lymphocyte (TIL) assessment in breast cancer",
    author="Yoni Schirris",
    author_email="yschirris@gmail.com",
    url="https://github.com/YoniSchirris/ectil",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
