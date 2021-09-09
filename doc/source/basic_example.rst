Demo 0: Example and usage
-------------------------

In order to make things simple the following rules have been followed during development:

* ``deel-lip`` follows the ``keras`` package structure.
* All elements (layers, activations, initializers, ...) are compatible with standard the ``keras`` elements.
* When a k-Lipschitz layer overrides a standard keras layer, it uses the same interface and the same parameters.
  The only difference is a new parameter to control the Lipschitz constant of a layer.

Which layers are safe to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~

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

Please note that when learning with the :class:`.HKR_loss` and :class:`.HKR_multiclass_loss`, no activation is
required on the last layer.


.. include:: demo0.rst
