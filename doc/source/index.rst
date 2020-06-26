.. keras lipschitz layers documentation master file, created by
   sphinx-quickstart on Mon Feb 17 16:42:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to deel-lip documentation!
==================================================

Controlling the Lipschitz constant of a layer or a whole neural network has many applications ranging
from adversarial robustness to Wasserstein distance estimation.

This library provides implementation of **k-Lispchitz layers for** ``keras``.

The library contains:
---------------------

 * k-Lipschitz variant of keras layers such as `Dense`, `Conv2D` and `Pooling`,
 * activation functions compatible with `keras`,
 * kernel initializers and kernel constraints for `keras`,
 * loss functions when working with Wasserstein distance estimations,
 * tools to monitor the singular values of kernels during training,
 * tools to convert k-Lipschitz network to regular network for faster evaluation.


Installation
------------

You can install ``deel-lip`` directly from pypi:

.. code-block:: bash

   pip install deel-lip

In order to use ``deel-lip``, you also need a `valid tensorflow installation <https://www.tensorflow.org/install>`_.
``deel-lip`` supports tensorflow from 2.0 to 2.2.

Cite this work
--------------

.. raw:: html

   This library has been built to support the work presented in the paper
   <a href="https://arxiv.org/abs/2006.06520"><i>Achieving robustness in classification using optimal transport with Hinge regularization</i></a>.
   This work can be cited as:

.. code-block:: latex

   @misc{2006.06520,
   Author = {
      Mathieu Serrurier
      and Franck Mamalet
      and Alberto González-Sanz
      and Thibaut Boissin
      and Jean-Michel Loubes
      and Eustasio del Barrio
   },
   Title = {
      Achieving robustness in classification using optimal transport with hinge regularization
   },
   Year = {2020},
   Eprint = {arXiv:2006.06520},
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :titlesonly:
   :maxdepth: 4
   :caption: Contents:
   :glob:

   basic_example.rst
   wasserstein_toy.rst
   wassersteinClassif_toy.rst
   wassersteinClassif_MNIST08.rst

   deel.lip
