# deel-lip

[![Python](https://img.shields.io/pypi/pyversions/deel-lip.svg)](https://pypi.org/project/deel-lip)
[![PyPI](https://img.shields.io/pypi/v/deel-lip.svg)](https://pypi.org/project/deel-lip)
[![Downloads](https://pepy.tech/badge/deel-lip)](https://pepy.tech/project/deel-lip)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://deel-lip.readthedocs.io)
[![deel-lip tests](https://github.com/deel-ai/deel-lip/actions/workflows/python-tests.yml/badge.svg?branch=master)](https://github.com/deel-ai/deel-lip/actions/workflows/python-tests.yml)
[![deel-lip linters](https://github.com/deel-ai/deel-lip/actions/workflows/python-linters.yml/badge.svg?branch=master)](https://github.com/deel-ai/deel-lip/actions/workflows/python-linters.yml)
[![GitHub license](https://img.shields.io/github/license/deel-ai/deel-lip.svg)](https://github.com/deel-ai/deel-lip/blob/master/LICENSE)

Controlling the Lipschitz constant of a layer or a whole neural network has many applications ranging
from adversarial robustness to Wasserstein distance estimation.

This library provides an efficient implementation of **k-Lispchitz layers for `keras`**.

## The library contains:

 * k-Lipschitz variant of keras layers such as `Dense`, `Conv2D` and `Pooling`,
 * activation functions compatible with `keras`,
 * kernel initializers and kernel constraints for `keras`,
 * loss functions that make use of Lipschitz constrained networks (see [our paper](https://arxiv.org/abs/2006.06520) for more information),
 * tools to monitor the singular values of kernels during training,
 * tools to convert k-Lipschitz network to regular network for faster inference.

## Example and usage

In order to make things simple the following rules have been followed during development:
* `deel-lip` follows the `keras` package structure.
* All elements (layers, activations, initializers, ...) are compatible with standard the `keras` elements.
* When a k-Lipschitz layer overrides a standard keras layer, it uses the same interface and the same parameters.
  The only difference is a new parameter to control the Lipschitz constant of a layer.

Here is an example showing how to build and train a 1-Lipschitz network:
```python
from deel.lip.layers import (
    SpectralDense,
    SpectralConv2D,
    ScaledL2NormPooling2D,
    FrobeniusDense,
)
from deel.lip.model import Sequential
from deel.lip.activations import GroupSort
from deel.lip.losses import MulticlassHKR, MulticlassKR
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Sequential (resp Model) from deel.model has the same properties as any lipschitz model.
# It act only as a container, with features specific to lipschitz
# functions (condensation, vanilla_exportation...) but The layers are fully compatible
# with the tf.keras.model.Sequential/Model
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
    loss=MulticlassHKR(alpha=50, min_margin=0.05),
    optimizer=Adam(1e-3),
    metrics=["accuracy", MulticlassKR()],
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
    batch_size=2048,
    epochs=30,
    validation_data=(x_test, y_test),
    shuffle=True,
)

# once training is finished you can convert
# SpectralDense layers into Dense layers and SpectralConv2D into Conv2D
# which optimize performance for inference
vanilla_model = model.vanilla_export()
```

See [the full documentation](https://deel-lip.readthedocs.io) for a complete API description.

## Installation

You can install ``deel-lip`` directly from pypi:
```bash
pip install deel-lip
```

In order to use `deel-lip`, you also need a [valid tensorflow installation](https://www.tensorflow.org/install).
`deel-lip` supports tensorflow versions 2.x

## Cite this work

This library has been built to support the work presented in the paper
[*Achieving robustness in classification using optimaltransport with Hinge regularization*](https://arxiv.org/abs/2006.06520)
which aim provable and efficient robustness by design.

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
