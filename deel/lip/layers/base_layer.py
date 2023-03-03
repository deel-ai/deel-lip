# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
This module extends original keras layers, in order to add k lipschitz constraint via
reparametrization. Currently, are implemented:
* Dense layer:
    as SpectralDense (and as FrobeniusDense when the layer has a single
    output)
* Conv2D layer:
    as SpectralConv2D (and as FrobeniusConv2D when the layer has a single
    output)
* AveragePooling:
    as ScaledAveragePooling
* GlobalAveragePooling2D:
    as ScaledGlobalAveragePooling2D
By default the layers are 1 Lipschitz almost everywhere, which is efficient for
wasserstein distance estimation. However for other problems (such as adversarial
robustness) the user may want to use layers that are at most 1 lipschitz, this can
be done by setting the param `eps_bjorck=None`.
"""

import abc


class LipschitzLayer(abc.ABC):
    """
    This class allows to set Lipschitz factor of a layer. Lipschitz layer must inherit
    this class to allow user to set the Lipschitz factor.
    Warning:
         This class only regroups useful functions when developing new Lipschitz layers.
         But it does not ensure any property about the layer. This means that
         inheriting from this class won't ensure anything about the Lipschitz constant.
    """

    k_coef_lip = 1.0
    """variable used to store the lipschitz factor"""
    coef_lip = None
    """
    define correction coefficient (ie. Lipschitz bound ) of the layer
    ( multiply the output of the layer by this constant )
    """

    def set_klip_factor(self, klip_factor):
        """
        Allow to set the Lipschitz factor of a layer.
        Args:
            klip_factor (float): the Lipschitz factor the user want to ensure.
        Returns:
            None
        """
        self.k_coef_lip = klip_factor

    @abc.abstractmethod
    def _compute_lip_coef(self, input_shape=None):
        """
        Some layers (like convolution) cannot ensure a strict Lipschitz constant (as
        the Lipschitz factor depends on the input data). Those layers then rely on the
        computation of a bounding factor. This function allows to compute this factor.
        Args:
            input_shape: the shape of the input of the layer.
        Returns:
            the bounding factor.
        """
        pass

    def _init_lip_coef(self, input_shape):
        """
        Initialize the Lipschitz coefficient of a layer.
        Args:
            input_shape: the layers input shape
        Returns:
            None
        """
        self.coef_lip = self._compute_lip_coef(input_shape)

    def _get_coef(self):
        """
        Returns:
            the multiplicative coefficient to be used on the result in order to ensure
            k-Lipschitzity.
        """
        if self.coef_lip is None:
            raise RuntimeError("compute_coef must be called before calling get_coef")
        return self.coef_lip * self.k_coef_lip


class Condensable(abc.ABC):
    """
    Some Layers don't optimize directly the kernel, this means that the kernel stored
    in the layer is not the kernel used to make predictions (called W_bar), To address
    this, these layers can implement the condense() function that make self.kernel equal
    to W_bar.
    This operation also allows to turn the Lipschitz layer to its keras equivalent e.g.
    The Dense layer that have the same predictions as the trained SpectralDense.
    """

    @abc.abstractmethod
    def condense(self):
        """
        The condense operation allows to overwrite the kernel and ensure that other
        variables are still consistent.
        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def vanilla_export(self):
        """
        This operation allows to turn this Layer to its super type, easing storage and
        serving.
        Returns:
             self as super type
        """
        pass
