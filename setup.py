# © IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All rights reserved. DEEL is a research
# program operated by IVADO, IRT Saint Exupéry, CRIAQ and ANITI - https://www.deel.ai/
import setuptools

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

dev_requires = [
    "black",
    "flake8",
    "flake8-black",
    "numpy",
]


setuptools.setup(
    name="deel-lip",
    version="1.0.0",
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
    install_requires=["numpy", "tensorflow>=2,<=2.2"],
    license="MIT",
    extras_require={"dev": dev_requires,},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
