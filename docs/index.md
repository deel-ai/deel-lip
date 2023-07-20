<div align="center">
    <img src="./assets/logo_white.svg#only-dark" width="75%" alt="DEEL-LIP" align="center" />
    <img src="./assets/logo.svg#only-light" width="75%" alt="DEEL-LIP" align="center" />
</div>
<br>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/pypi/pyversions/deel-lip.svg">
    </a>
    <a href="https://github.com/deel-ai/deel-lip/actions/workflows/python-linters.yml">
        <img alt="PyLint" src="https://github.com/deel-ai/deel-lip/actions/workflows/python-linters.yml/badge.svg?branch=master">
    </a>
    <a href="https://github.com/deel-ai/deel-lip/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/deel-ai/deel-lip/actions/workflows/python-linters.yml/badge.svg?branch=master">
    </a>
    <a href="https://pypi.org/project/deel-lip">
        <img alt="Pypi" src="https://img.shields.io/pypi/v/deel-lip.svg">
    </a>
    <a href="https://pepy.tech/project/deel-lip">
        <img alt="Pepy" src="https://pepy.tech/badge/deel-lip">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
    <br>
    <a href="https://deel-ai.github.io/deel-lip/"><strong>Explore DEEL-LIP docs »</strong></a>
</div>
<br>

## 👋 Welcome to deel-lip documentation!

Controlling the Lipschitz constant of a layer or a whole neural network
has many applications ranging from adversarial robustness to Wasserstein
distance estimation.

This library provides an efficient implementation of **k-Lispchitz
layers for keras**.

## 📚 Table of contents

- [📚 Table of contents](#-table-of-contents)
- [🔥 Tutorials](#-tutorials)
- [🚀 Quick Start](#-quick-start)
- [📦 What's Included](#-whats-included)
- [👍 Contributing](#-contributing)
- [👀 See Also](#-see-also)
- [🙏 Acknowledgments](#-acknowledgments)
- [🗞️ Citation](#-citation)
- [📝 License](#-license)

## 🚀 Quick Start

You can install ``deel-lip`` directly from pypi:

```python
pip install deel-lip
```

In order to use ``deel-lip``, you also need a [valid tensorflow
installation](https://www.tensorflow.org/install). ``deel-lip``
supports tensorflow versions 2.x.

## 🔥 Tutorials

| **Tutorial Name**           | Notebook                                                                                                                                                           |
| :-------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Getting Started             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/demo0.ipynb)            |
| Wasserstein distance estimation on toy example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/demo1.ipynb) |
| HKR Classifier on toy dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/demo2.ipynb) |
| HKR classifier on MNIST dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/demo3.ipynb) |
| HKR multiclass and fooling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/demo4.ipynb) |


## 📦 What's Included

*  k-Lipschitz variant of keras layers such as ``Dense``, ``Conv2D`` and
   ``Pooling``,
*  activation functions compatible with ``keras``,
*  kernel initializers and kernel constraints for ``keras``,
*  loss functions that make use of Lipschitz constrained networks (see
   [our paper](https://arxiv.org/abs/2006.06520) for more
   information),
*  tools to monitor the singular values of kernels during training,
*  tools to convert k-Lipschitz network to regular network for faster
   inference.

## 👍 Contributing

To contribute, you can open an
[issue](https://github.com/deel-ai/deel-lip/issues), or fork this
repository and then submit changes through a
[pull-request](https://github.com/deel-ai/deel-lip/pulls).
We use [black](https://pypi.org/project/black/) to format the code and follow PEP-8 convention.
To check that your code will pass the lint-checks, you can run:

```python
tox -e py36-lint
```

You need [`tox`](https://tox.readthedocs.io/en/latest/) in order to
run this. You can install it via `pip`:

```python
pip install tox
```

## 👀 See Also

More from the DEEL project:

- [Xplique](https://github.com/deel-ai/xplique) a Python library exclusively dedicated to explaining neural networks.
- [Influenciae](https://github.com/deel-ai/influenciae) Python toolkit dedicated to computing influence values for the discovery of potentially problematic samples in a dataset.
- [deel-torchlip](https://github.com/deel-ai/deel-torchlip) a Python library for training k-Lipschitz neural networks on PyTorch.
- [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of data quality, representativity and explainability for this purpose.

## 🙏 Acknowledgments

<img align="right" src="https://share.deel.ai/apps/theming/image/logo?useSvg=1&v=10#only-dark" width="25%" alt="DEEL Logo" />
<img align="right" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png#only-light" width="25%" alt="DEEL Logo" />
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

## 🗞️ Citation

This library has been built to support the work presented in the paper
[Achieving robustness in classification using optimaltransport with
Hinge regularization](https://arxiv.org/abs/2006.06520) which aim
provable and efficient robustness by design.

This work can be cited as:

```
@misc{2006.06520,
    Author = {Mathieu Serrurier and Franck Mamalet and Alberto González-Sanz and Thibaut Boissin and Jean-Michel Loubes and Eustasio del Barrio},
    Title = {Achieving robustness in classification using optimal transport with hinge regularization},
    Year = {2020},
    Eprint = {arXiv:2006.06520},
}
```

## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
