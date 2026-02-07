"""Layer for application of ReLU to all elements except the ones for the boundary."""
import tensorflow as tf


class PartialReLU(tf.keras.layers.Layer):
    """Layer for application of ReLU to all elements except the ones for the boundary."""

    def __init__(self) -> None:
        """Initialize the layer."""
        super().__init__()
        self._minimum_length = 2

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply ReLU to all elements except the boundaries.

        Args:
            inputs: tf.Tensor of shape (features,) or (1, features)

        Returns:
            tf.Tensor of same shape, with ReLU applied to middle elements.
        """
        n = tf.shape(inputs)[1]

        def too_short() -> tf.Tensor:
            """Return tensor as-is."""
            return inputs

        def relu_middle() -> tf.Tensor:
            first = tf.cast(inputs[:, 0:1], tf.float64)
            last = tf.cast(inputs[:, -1:], tf.float64)

            middle = tf.nn.relu(inputs[:, 1:-1]) + tf.constant(10**-14, dtype=tf.float64)
            return tf.concat([first, middle, last], axis=1)

        return tf.cond(n <= self._minimum_length, too_short, relu_middle)
