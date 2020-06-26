# © IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All rights reserved. DEEL is a research
# program operated by IVADO, IRT Saint Exupéry, CRIAQ and ANITI - https://www.deel.ai/
"""
Contains utility functions.
"""
from typing import Generator, Tuple, Any
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import (
    load_model as keras_load_model,
    model_from_json as keras_model_from_json,
    model_from_yaml as keras_model_from_yaml,
)

CUSTOM_OBJECTS = dict()


def _deel_export(f):
    """
    Annotation, allows to automatically add deel custom objects to the deel.lip.utils.CUSTOM_OBJECTS variable, which
    is useful when working with custom layers.
    """
    global CUSTOM_OBJECTS
    CUSTOM_OBJECTS[f.__name__] = f
    return f


def load_model(filepath, custom_objects=None, compile=True) -> Model:
    """
    Equivalent to load_model from keras, but custom_objects are already known

    Args:
        filepath: One of the following: - String, path to the saved model - `h5py.File` object from which to load the
        model.
        custom_objects: Optional dictionary mapping names (strings) to custom classes or functions to be considered
        during deserialization.
        compile: Boolean, whether to compile the model after loading.

    Returns: A Keras model instance. If an optimizer was found

    """
    deel_custom_objects = CUSTOM_OBJECTS.copy()
    if custom_objects is not None:
        deel_custom_objects.update(custom_objects)
    return keras_load_model(filepath, deel_custom_objects, compile)


def model_from_json(json_string, custom_objects=None) -> Model:
    """
    Equivalent to model_from_json from keras, but custom_objects are already known.

    Args:
        json_string: JSON string encoding a model configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom classes or functions to be considered
        during deserialization.

    Returns: A Keras model instance (uncompiled).

    """
    deel_custom_objects = CUSTOM_OBJECTS.copy()
    if custom_objects is not None:
        deel_custom_objects.update(custom_objects)
    return keras_model_from_json(
        json_string=json_string, custom_objects=deel_custom_objects
    )


def model_from_yaml(yaml_string, custom_objects=None) -> Model:
    """
    Equivalent to model_from_json from keras, but custom_objects are already known

    Args:
        yaml_string: YAML string encoding a model configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom classes or functions to be considered
        during deserialization.

    Returns: A Keras model instance (uncompiled).

    """
    deel_custom_objects = CUSTOM_OBJECTS.copy()
    if custom_objects is not None:
        deel_custom_objects.update(custom_objects)
    return keras_model_from_yaml(
        yaml_string=yaml_string, custom_objects=deel_custom_objects
    )


def evaluate_lip_const_gen(
    model: Model,
    generator: Generator[Tuple[np.ndarray, np.ndarray], Any, None],
    eps=1e-4,
    seed=None,
):
    """
    Evaluate the Lipschitz constant of a model, with the naive method.
    Please note that the estimation of the lipschitz constant is done locally around input sample. This may not
    correctly estimate the behaviour in the whole domain. The computation might also be inaccurate in high dimensional
    space.

    This is the generator version of evaluate_lip_const.

    Args:
        model: built keras model used to make predictions
        generator: used to select datapoints where to compute the lipschitz constant
        eps: magnitude of noise to add to input in order to compute the constant
        seed: seed used when generating the noise ( can be set to None )

    Returns: the empirically evaluated lipschitz constant.

    """
    x, y = generator.send(None)
    return evaluate_lip_const(model, x, eps, seed=seed)


def evaluate_lip_const(model: Model, x, eps=1e-4, seed=None):
    """
    Evaluate the Lipschitz constant of a model, with the naive method.
    Please note that the estimation of the lipschitz constant is done locally around input sample. This may not
    correctly estimate the behaviour in the whole domain.

    Args:
        model: built keras model used to make predictions
        x: inputs used to compute the lipschitz constant
        eps: magnitude of noise to add to input in order to compute the constant
        seed: seed used when generating the noise ( can be set to None )

    Returns: the empirically evaluated lipschitz constant. The computation might also be inaccurate in high dimensional
    space.

    """
    y_pred = model.predict(x)
    # x = np.repeat(x, 100, 0)
    # y_pred = np.repeat(y_pred, 100, 0)
    x_var = x + K.random_uniform(
        shape=x.shape, minval=eps * 0.25, maxval=eps, seed=seed
    )
    y_pred_var = model.predict(x_var)
    dx = x - x_var
    dfx = y_pred - y_pred_var
    ndx = K.sum(K.square(dx), axis=range(1, len(x.shape)))
    ndfx = K.sum(K.square(dfx), axis=range(1, len(y_pred.shape)))
    lip_cst = K.sqrt(K.max(ndfx / ndx))
    print("lip cst: %.3f" % lip_cst)
    return lip_cst
