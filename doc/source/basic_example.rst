Example and usage
=================


In order to make things simple the following rules have been followed during development:

* ``deel-lip`` follows the ``keras`` package structure.
* All elements (layers, activations, initializers, ...) are compatible with standard the ``keras`` elements.
* When a k-Lipschitz layer overrides a standard keras layer, it uses the same interface and the same parameters.
  The only difference is a new parameter to control the Lipschitz constant of a layer.

Which layers are safe to use?
-----------------------------

The following table indicates which layers are safe to use in a Lipshitz network, and which are not.

.. role:: raw-html-m2r(raw)
   :format: html


.. list-table::
   :header-rows: 1

   * - layer
     - 1-lip?
     - deel-lip equivalent
     - comments
   * - :class:`Dense`
     - no
     - :class:`.SpectralDense` \ :raw-html-m2r:`<br>`\ :class:`.FrobeniusDense`
     - :class:`.SpectralDense` and :class:`.FrobeniusDense` are similar when there is a single output.
   * - :class:`Conv2D`
     - no
     - :class:`.SpectralConv2D` \ :raw-html-m2r:`<br>`\ :class:`.FrobeniusConv2D`
     - :class:`.SpectralConv2D` also implements Björck normalization.
   * - :class:`MaxPooling`\ :raw-html-m2r:`<br>`\ :class:`GlobalMaxPooling`
     - yes
     - n/a
     -
   * - :class:`AveragePooling2D`\ :raw-html-m2r:`<br>`\ :class:`GlobalAveragePooling2D`
     - no
     - :class:`.ScaledAveragePooling2D`\ :raw-html-m2r:`<br>`\ :class:`.ScaledGlobalAveragePooling2D`
     - The lipschitz constant is bounded by ``sqrt(pool_h * pool_h)``.
   * - :class:`Flatten`
     - yes
     - n/a
     -
   * - :class:`Dropout`
     - no
     - None
     - The lipschitz constant is bounded by the dropout factor.
   * - :class:`BatchNorm`
     - no
     - None
     - We suspect that layer normalization already limits internal covariate shift.

Design tips
-----------

Designing lipschitz networks require a careful design in order to avoid vanishing/exploding gradient problem.

Choosing pooling layers:

.. role:: raw-html-m2r(raw)
   :format: html

.. list-table::
   :header-rows: 1

   * - layer
     - advantages
     - disadvantages
   * - :class:`.ScaledAveragePooling2D` and :class:`.MaxPooling2D`
     - very similar to original implementation (just add a scaling factor for avg).
     - not norm preserving nor gradient norm preserving.
   * - :class:`.InvertibleDownSampling`
     - norm preserving and gradient norm preserving.
     - increases the number of channels (and the number of parameters of the next layer).
   * - :class:`.ScaledL2NormPooling2D` ( `sqrt(avgpool(x**2))` )
     - norm preserving.
     - lower numerical stability of the gradient when inputs are close to zero.


Choosing activations:


.. role:: raw-html-m2r(raw)
   :format: html

.. list-table::
   :header-rows: 1

   * - layer
     - advantages
     - disadvantages
   * - :class:`ReLU`
     -
     - create a strong vanishing gradient effect. If you manage to learn with it, please call 911.
   * - :class:`.MaxMin` (`stack([ReLU(x), ReLU(-x)])`)
     - have similar properties to ReLU, but is norm and gradient norm preserving
     - double the number of outputs
   * - :class:`.GroupSort`
     - Input and GradientNorm preserving. Also limit the need of biases (as it is shift invariant).
     - more computationally expensive, (when it's parameter `n` is large)

Please note that when learning with the :class:`.HKR` and :class:`.MulticlassHKR`, no
activation is
required on the last layer.

How to use it?
--------------

Here is an example showing how to build and train a 1-Lipschitz network:

.. code:: python
    from deel.lip.layers import (
        SpectralDense,
        SpectralConv2D,
        ScaledL2NormPooling2D,
        FrobeniusDense,
    )
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
                use_bias=True,
                kernel_initializer="orthogonal",
            ),
            # usual pooling layer are implemented (avg, max...), but new layers are also available
            ScaledL2NormPooling2D(pool_size=(2, 2), data_format="channels_last"),
            SpectralConv2D(
                filters=16,
                kernel_size=(3, 3),
                activation=GroupSort(2),
                use_bias=True,
                kernel_initializer="orthogonal",
            ),
            ScaledL2NormPooling2D(pool_size=(2, 2), data_format="channels_last"),
            # our layers are fully interoperable with existing keras layers
            Flatten(),
            SpectralDense(
                32,
                activation=GroupSort(2),
                use_bias=True,
                kernel_initializer="orthogonal",
            ),
            FrobeniusDense(
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
        # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
        # note also in the case of lipschitz networks, more robustness require more parameters.
        loss=HKR_multiclass_loss(alpha=25, min_margin=0.25),
        optimizer=Adam(lr=0.005),
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
