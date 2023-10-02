# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The submodule `deel.lip.layers` contains all custom Keras layers to build
Lipschitz-constrained neural networks. They all inherit from `keras.layers.Layer` from
Keras API.
"""
from . import unconstrained
from .activations import FullSort, GroupSort, GroupSort2, Householder, MaxMin, PReLUlip
from .base_layer import Condensable, LipschitzLayer
from .convolutional import FrobeniusConv2D, SpectralConv2D, SpectralConv2DTranspose
from .dense import FrobeniusDense, SpectralDense
from .pooling import (
    InvertibleDownSampling,
    InvertibleUpSampling,
    ScaledAveragePooling2D,
    ScaledGlobalAveragePooling2D,
    ScaledGlobalL2NormPooling2D,
    ScaledL2NormPooling2D,
)
