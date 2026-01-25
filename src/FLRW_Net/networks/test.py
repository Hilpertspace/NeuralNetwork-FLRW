"""Minimum example for a generalized model."""

import tensorflow as tf

from FLRW_Net.layers.matmul import Matmul
from FLRW_Net.layers.relu import PartialReLU


class GeneralizedModel(tf.keras.Model):
    """Minimum example for a generalized model."""

    def __init__(self, slice_specs: list[tuple[int, int]]) -> None:
        """Initialize the model."""
        super().__init__()
        self.slice_layers = [Matmul(start, end) for start, end in slice_specs]
        self.relu_layer = PartialReLU()

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Call the model."""
        outputs_1 = [layer(inputs) for layer in self.slice_layers]
        outputs_1 = tf.concat(outputs_1, axis=1)

        outputs = self.relu_layer(outputs_1)

        return tf.concat(outputs, axis=-1)
