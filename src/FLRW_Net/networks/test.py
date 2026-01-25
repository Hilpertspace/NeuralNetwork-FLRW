"""Minimum example for a generalized model."""

import tensorflow as tf

from FLRW_Net.layers.matmul import Matmul
from FLRW_Net.layers.relu import PartialReLU
from FLRW_Net.layers.strut_activation import StrutActivation


class GeneralizedModel(tf.keras.Model):
    """Minimum example for a generalized model."""

    def __init__(self, slice_specs: list[tuple[int, int]]) -> None:
        """Initialize the model."""
        super().__init__()
        self.slice_layers = [Matmul(start, end) for start, end in slice_specs]
        self.relu_layer = PartialReLU()
        self.strut_activation_layer = StrutActivation()

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Call the model."""
        outputs_1 = [layer(inputs) for layer in self.slice_layers]
        outputs_1 = tf.concat(outputs_1, axis=1)

        outputs_2 = self.relu_layer(outputs_1)
        outputs_3 = self.strut_activation_layer(outputs_2)

        return tf.concat(outputs_3, axis=-1)
