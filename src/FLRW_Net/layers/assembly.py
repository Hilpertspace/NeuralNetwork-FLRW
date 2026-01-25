"""Layer for assembling the input for the final strut activation."""
import tensorflow as tf


class Assembly(tf.keras.layers.Layer):
    """Layer for assembling the input for the final strut activation."""

    def __init__(self) -> None:
        """Initialize the layer."""
        super().__init__()

    @tf.function
    def call(self, outputs_2: tf.Tensor, outputs_4: tf.Tensor) -> tf.Tensor:
        """Assemble the input from scaled a's and the updates spatial edges.

        Alternate values from outputs_2 and outputs_4.

        Pattern:
        [boundary, input_1, input_2, input_1, input_2, ..., boundary]

        Args:
            outputs_2: tf.Tensor containing the scaled a's (the updated struts before activation)
            outputs_4: tf.Tensor containing the updated spatial edges

        Returns:
            tf.Tensor of shape [1, n]
        """
        n: tf.Tensor = tf.shape(outputs_2)[1]
        output: tf.Tensor = tf.identity(outputs_2)

        odd_indices: tf.Tensor = tf.range(1, n-1, 2)
        even_indices: tf.Tensor = tf.range(2, n-1, 2)

        scaled_as: tf.Tensor = tf.gather(outputs_2, odd_indices, axis=1)
        updated_spatial_edges: tf.Tensor = tf.gather(outputs_4, even_indices, axis=1)

        # Scatter odd values from input_1
        odd_scatter_indices = tf.stack([tf.zeros_like(odd_indices), odd_indices], axis=1)
        output = tf.tensor_scatter_nd_update(output, odd_scatter_indices, tf.reshape(scaled_as, [-1]))

        # Scatter even values from input_2
        even_scatter_indices = tf.stack([tf.zeros_like(even_indices), even_indices], axis=1)
        output = tf.tensor_scatter_nd_update(output, even_scatter_indices, tf.reshape(updated_spatial_edges, [-1]))

        return output
