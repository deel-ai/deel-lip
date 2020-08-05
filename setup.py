# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import setuptools

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

dev_requires = [
    "tox",
    "black",
    "flake8",
    "flake8-black",
    "numpy",
]

docs_requires = [
    "sphinx",
    "recommonmark",
    "sphinx_rtd_theme",
    "sphinx_markdown_builder",
    "ipython",  # required for Pygments
]

setuptools.setup(
    name="deel-lip",
    version="1.0.1",
    author=", ".join(["Mathieu SERRURIER", "Franck MAMALET", "Thibaut BOISSIN"]),
    author_email=", ".join(
        [
            "mathieu.serrurier@irt-saintexupery.com",
            "franck.mamalet@irt-saintexupery.com",
            "thibaut.boissin@irt-saintexupery.com",
        ]
    ),
    description="TensorFlow 2 implementation for k-Lipschitz layers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deel-ai/deel-lip",
    packages=setuptools.find_namespace_packages(include=["deel.*"]),
    install_requires=["numpy", "tensorflow>=2,<=2.3"],
    license="MIT",
    extras_require={"dev": dev_requires, "docs": docs_requires},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
