# deel-lip

[![Python](https://img.shields.io/pypi/pyversions/deel-lip.svg)](https://pypi.org/project/deel-lip)
[![PyPI](https://img.shields.io/pypi/v/deel-lip.svg)](https://pypi.org/project/deel-lip)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://deel-lip.readthedocs.io)
[![GitHub license](https://img.shields.io/github/license/deel-ai/deel-lip.svg)](https://github.com/deel-ai/deel-lip/blob/master/LICENSE)

Controlling the Lipschitz constant of a layer or a whole neural network has many applications ranging
from adversarial robustness to Wasserstein distance estimation.

This library provides implementation of **k-Lispchitz layers for `keras`**.

## The library contains:

 * k-Lipschitz variant of keras layers such as `Dense`, `Conv2D` and `Pooling`,
 * activation functions compatible with `keras`,
 * kernel initializers and kernel constraints for `keras`,
 * loss functions when working with Wasserstein distance estimations,
 * tools to monitor the singular values of kernels during training,
 * tools to convert k-Lipschitz network to regular network for faster evaluation.

## Example and usage

In order to make things simple the following rules have been followed during development:
* `deel-lip` follows the `keras` package structure.
* All elements (layers, activations, initializers, ...) are compatible with standard the `keras` elements.
* When a k-Lipschitz layer overrides a standard keras layer, it uses the same interface and the same parameters.
  The only difference is a new parameter to control the Lipschitz constant of a layer.

Here is a simple example showing how to build a 1-Lipschitz network:
```python
from deel.lip.initializers import BjorckInitializer
from deel.lip.layers import SpectralDense, SpectralConv2D, ScaledL2NormPooling2D
from deel.lip.model import Model
from deel.lip.activations import PReLUlip
from tensorflow.keras.layers import Input, Lambda, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

# Sequential (resp Model) from deel.model has the same properties as any lipschitz
# layer ( condense, setting of the lipschitz factor etc...). It act only as a container.
model = Model(
    [
        Input(shape=(28, 28)),
        Lambda(lambda x: K.reshape(x, (-1, 28, 28, 1))),

        # Lipschitz layer preserve the API of their superclass ( here Conv2D )
        # an optional param is available: k_coef_lip which control the lipschitz
        # constant of the layer
        SpectralConv2D(
            filters=32, kernel_size=(3, 3), padding='same', activation=PReLUlip(),
            input_shape=(28, 28, 1), data_format='channels_last',
             kernel_initializer=BjorckInitializer(15, 50)),
        SpectralConv2D(
            filters=32, kernel_size=(3, 3), padding='same', activation=PReLUlip(),
            input_shape=(28, 28, 1), data_format='channels_last',
             kernel_initializer=BjorckInitializer(15, 50)),
        ScaledL2NormPooling2D(pool_size=(2, 2), data_format='channels_last'),

        SpectralConv2D(
            filters=64, kernel_size=(3, 3), padding='same', activation=PReLUlip(),
            input_shape=(28, 28, 1), data_format='channels_last',
             kernel_initializer=BjorckInitializer(15, 50)),
        SpectralConv2D(
            filters=64, kernel_size=(3, 3), padding='same', activation=PReLUlip(),
            input_shape=(28, 28, 1), data_format='channels_last',
             kernel_initializer=BjorckInitializer(15, 50)),
        ScaledL2NormPooling2D(pool_size=(2, 2), data_format='channels_last'),

        Flatten(),
        SpectralDense(256, activation="relu", kernel_initializer=BjorckInitializer(15, 50)),
        SpectralDense(10, activation="softmax"),
    ],
    k_coef_lip=0.5,
    name='testing'
)

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
```

See [the full documentation](https://deel-lip.readthedocs.io) for a complete API description.

## Installation

You can install ``deel-lip`` directly from pypi:
```bash
pip install deel-lip
```

In order to use `deel-lip`, you also need a [valid tensorflow installation](https://www.tensorflow.org/install).
`deel-lip` supports tensorflow from 2.0 to 2.2.

## Cite this work

This library has been built to support the work presented in the paper
*Achieving robustness in classification using optimaltransport with Hinge regularization*.

This work can be cited as:
```latex
@misc{2006.06520,
Author = {Mathieu Serrurier and Franck Mamalet and Alberto González-Sanz and Thibaut Boissin and Jean-Michel Loubes and Eustasio del Barrio},
Title = {Achieving robustness in classification using optimal transport with hinge regularization},
Year = {2020},
Eprint = {arXiv:2006.06520},
}
```

## Contributing

To contribute, you can open an [issue](https://github.com/deel-ai/deel-lip/issues), or fork this repository and then submit
changes through a [pull-request](https://github.com/deel-ai/deel-lip/pulls).
We use [`black`](https://pypi.org/project/black/) to format the code and follow PEP-8 convention. To check
that your code will pass the lint-checks, you can run:

```bash
tox -e py36-lint
```

You need [`tox`](https://tox.readthedocs.io/en/latest/) in order to run this. You can install it via `pip`:

```bash
pip install tox
```

## License

Copyright 2020 © IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry, CRIAQ and ANITI - https://www.deel.ai/

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgments

This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and
Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the [DEEL
project](https://www.deel.ai/).
