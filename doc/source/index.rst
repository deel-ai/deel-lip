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

Example and usage
-----------------

In order to make things simple the following rules have been followed
during development:
- ``deel-lip`` follows the ``keras`` package structure.
- All elements (layers, activations, initializers, ...) are compatible with standard the ``keras`` elements.
- When a k-Lipschitz layer overrides a standard keras layer, it uses the same interface and
the same parameters. The only difference is a new parameter to control the Lipschitz constant of a layer.

Here is an example showing how to build and train a 1-Lipschitz network:

.. code:: python

    from deel.lip.layers import SpectralDense, SpectralConv2D, ScaledL2NormPooling2D
    from deel.lip.model import Sequential
    from deel.lip.activations import GroupSort
    from deel.lip.losses import HKR_multiclass_loss
    from tensorflow.keras.layers import Input, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    import numpy as np

    # Sequential (resp Model) from deel.model has the same properties as any lipschitz model.
    # It act only as a container, with features specific to lipschitz
    # functions (condensation, vanilla_exportation...)
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            # Lipschitz layers preserve the API of their superclass ( here Conv2D )
            # an optional param is available: k_coef_lip which control the lipschitz
            # constant of the layer
            SpectralConv2D(
                filters=16,
                kernel_size=(3, 3),
                activation=GroupSort(2),
                use_bias=False,
                kernel_initializer="orthogonal",
            ),
            # usual pooling layer are implemented (avg, max...), but new layers are also available
            ScaledL2NormPooling2D(pool_size=(2, 2), data_format="channels_last"),
            SpectralConv2D(
                filters=32,
                kernel_size=(3, 3),
                activation=GroupSort(2),
                use_bias=False,
                kernel_initializer="orthogonal",
            ),
            ScaledL2NormPooling2D(pool_size=(2, 2), data_format="channels_last"),
            # our layers are fully interoperable with existing keras layers
            Flatten(),
            SpectralDense(
                100,
                activation=GroupSort(2),
                use_bias=False,
                kernel_initializer="orthogonal",
            ),
            SpectralDense(
                10, activation=None, use_bias=False, kernel_initializer="orthogonal"
            ),
        ],
        # similary model has a parameter to set the lipschitz constant
        # to set automatically the constant of each layer
        k_coef_lip=1.0,
        name="hkr_model",
    )

    # HKR (Hinge-Krantorovich-Rubinstein) optimize robustness along with accuracy
    model.compile(
        loss=HKR_multiclass_loss(alpha=5.0, min_margin=0.5),
        optimizer=Adam(lr=0.01),
        metrics=["accuracy"],
    )

    model.summary()

    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # standardize and reshape the data
    x_train = np.expand_dims(x_train, -1)
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = np.expand_dims(x_test, -1)
    x_test = (x_test - mean) / std
    # one hot encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # fit the model
    model.fit(
        x_train,
        y_train,
        batch_size=256,
        epochs=15,
        validation_data=(x_test, y_test),
        shuffle=True,
    )

    # once training is finished you can convert
    # SpectralDense layers into Dense layers and SpectralConv2D into Conv2D
    # which optimize performance for inference
    vanilla_model = model.vanilla_export()

See `the full documentation <https://deel-lip.readthedocs.io>`__ for a
complete API description.

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
   wasserstein_toy.rst
   wassersteinClassif_toy.rst
   wassersteinClassif_MNIST08.rst

   deel.lip
