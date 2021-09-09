.. keras lipschitz layers documentation master file, created by
   sphinx-quickstart on Mon Feb 17 16:42:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to deel-lip documentation!
==================================================
Controlling the Lipschitz constant of a layer or a whole neural network
has many applications ranging from adversarial robustness to Wasserstein
distance estimation.

This library provides an efficient implementation of **k-Lispchitz
layers for keras**.

The library contains:
---------------------

-  k-Lipschitz variant of keras layers such as ``Dense``, ``Conv2D`` and
   ``Pooling``,
-  activation functions compatible with ``keras``,
-  kernel initializers and kernel constraints for ``keras``,
-  loss functions that make use of Lipschitz constrained networks (see
   `our paper <https://arxiv.org/abs/2006.06520>`__ for more
   information),
-  tools to monitor the singular values of kernels during training,
-  tools to convert k-Lipschitz network to regular network for faster
   inference.


Installation
------------

You can install ``deel-lip`` directly from pypi:

.. code:: bash

    pip install deel-lip

In order to use ``deel-lip``, you also need a `valid tensorflow
installation <https://www.tensorflow.org/install>`__. ``deel-lip``
supports tensorflow versions 2.x

Cite this work
--------------

This library has been built to support the work presented in the paper
`Achieving robustness in classification using optimaltransport with
Hinge regularization <https://arxiv.org/abs/2006.06520>`__ which aim
provable and efficient robustness by design.

This work can be cited as:

.. code:: latex

    @misc{2006.06520,
    Author = {Mathieu Serrurier and Franck Mamalet and Alberto González-Sanz and Thibaut Boissin and Jean-Michel Loubes and Eustasio del Barrio},
    Title = {Achieving robustness in classification using optimal transport with hinge regularization},
    Year = {2020},
    Eprint = {arXiv:2006.06520},
    }

Contributing
------------

To contribute, you can open an
`issue <https://github.com/deel-ai/deel-lip/issues>`__, or fork this
repository and then submit changes through a
`pull-request <https://github.com/deel-ai/deel-lip/pulls>`__.
We use `black <https://pypi.org/project/black/>`__ to format the code and follow PEP-8 convention.
To check that your code will pass the lint-checks, you can run:

.. code:: bash

    tox -e py36-lint

You need `tox <https://tox.readthedocs.io/en/latest/>`__ in order to
run this. You can install it via ``pip``:

.. code:: bash

    pip install tox


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
   demo1.rst
   demo2.rst
   demo3.rst
   demo4.rst

   deel.lip
