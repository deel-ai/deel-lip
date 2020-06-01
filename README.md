# deel-lip

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
from deel.lip.layers import ScaledMaxPooling2D, SpectralDense, SpectralConv2D
from deel.lip.model import Model
from deel.lip.activations import PReLUlip
from tensorflow.keras.layers import Input, Lambda, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

# Sequential (resp Model) from deel.model has the same properties as any lipschitz layer ( condense,
# setting of the lipschitz factor etc...). It act only as a container.
model = Model(
    [
        Input(shape=(28, 28)),
        Lambda(lambda x: K.reshape(x, (-1, 28, 28, 1))),

        # Lipschitz layer preserve the API of their superclass ( here Conv2D )
        # an optional param is available: k_coef_lip which control the lipschitz constant of the layer
        SpectralConv2D(
            filters=32, kernel_size=(3, 3), padding='same', activation=PReLUlip(), input_shape=(28, 28, 1),
            data_format='channels_last', kernel_initializer=BjorckInitializer(15, 50)),
        SpectralConv2D(
            filters=32, kernel_size=(3, 3), padding='same', activation=PReLUlip(), input_shape=(28, 28, 1),
            data_format='channels_last', kernel_initializer=BjorckInitializer(15, 50)),
        ScaledMaxPooling2D(pool_size=(2, 2), data_format='channels_last'),

        SpectralConv2D(
            filters=64, kernel_size=(3, 3), padding='same', activation=PReLUlip(), input_shape=(28, 28, 1),
            data_format='channels_last', kernel_initializer=BjorckInitializer(15, 50)),
        SpectralConv2D(
            filters=64, kernel_size=(3, 3), padding='same', activation=PReLUlip(), input_shape=(28, 28, 1),
            data_format='channels_last', kernel_initializer=BjorckInitializer(15, 50)),
        ScaledMaxPooling2D(pool_size=(2, 2), data_format='channels_last'),

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

See [the full documentation](http://deel-ai.github.io/deel-lip) for a complete API description.

## Installation

You can install ``deel-lip`` directly from pypi: 
```bash
pip install deel-lip
```

In order to use `deel-lip`, you also need a [valid tensorflow installation](https://www.tensorflow.org/install). 
`deel-lip` supports tensorflow 2.0 and tensorflow 2.1.

## Cite this work

This library has been built to support the work presented in the paper 
*Achieving robustness in classification using optimaltransport with Hinge regularization*.

This work can be cited as:
````latex
@misc{deellip,
  title={Achieving robustness in classification using optimaltransport with Hinge regularization},
  author={Mathieu Serrurier, Franck Mamalet, Alberto Gonźalez-Sanz,Thibaut Boissin, Jean-Michel Loubes, Eustasio del Barrio},
  year={2020},
  organization={DEEL}
}
````

## License

Copyright 2020 DEEL Team

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgments

This project received funding from the French ”Investing for the Future – PIA3” program within the Artiﬁcial and 
Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the [DEEL 
project](https://www.deel.ai/).
