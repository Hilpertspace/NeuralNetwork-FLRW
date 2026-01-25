"""Minimum example for a generalized model."""

import tensorflow as tf
from FLRW_Net.layers.matmul import Matmul


class GeneralizedModel(tf.keras.Model):
    """Minimum example for a generalized model."""

    def __init__(self, slice_specs: list[tuple[int, int]]) -> None:
        """Initialize the model."""
        super().__init__()
        self.slice_layers = [Matmul(start, end) for start, end in slice_specs]

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Call the model."""
        outputs = [layer(inputs) for layer in self.slice_layers]

        return tf.concat(outputs, axis=-1)
