"""Layer for the application of the activation function for spatial edges."""
import tensorflow as tf


class SpatialEdgeActivation(tf.keras.layers.Layer):
    """Layer for the application of the activation function for spatial edges."""

    def __init__(self) -> None:
        """Initialize the layer."""
        super().__init__()

    @tf.function
    def _spatial_edge_activation(self, inputs: tf.Tensor) -> tf.Tensor:
        """Activation function for the spatial-edge neurons."""
        l_n = tf.cast(inputs[:, 0:1], tf.float64)
        m_n = tf.cast(inputs[:, 1:2], tf.float64)
        l_n_plus_one = tf.cast(inputs[:, 2:3], tf.float64)
        m_n_plus_one = tf.cast(inputs[:, 3:4], tf.float64)
        l_n_plus_two = tf.cast(inputs[:, 4:5], tf.float64)

        part_1 = l_n + (l_n_plus_two - l_n) * tf.constant(3 / 8, dtype=tf.float64) * tf.math.square(l_n - l_n_plus_one) / tf.math.square(m_n)  # noqa: E501
        part_2 = l_n + (l_n_plus_two - l_n) * tf.constant(3 / 8, dtype=tf.float64) * tf.math.square(l_n_plus_one - l_n_plus_two) / tf.math.square(m_n_plus_one)  # noqa: E501
        output = (part_1 + part_2) / tf.constant(2, dtype=tf.float64)

        return output

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply the strut activation function to the strut-neurons."""
        num_features = tf.shape(inputs)[1]

        # Indices of every third feature (2, 5, 8, ...)
        indices = tf.range(2, num_features-2, 2)

        # Compute start and end indices for 3-feature slices
        start_indices = indices - 2
        end_indices = indices + 3

        # Vectorized: stack start and end indices
        slice_indices = tf.stack([start_indices, end_indices], axis=1)

        # Gather all 5-feature slices in a vectorized way
        def gather_slice(i: tf.Tensor) -> tf.Tensor:
            return inputs[:, i[0]:i[1]]  # shape [1,5]

        # Gather all slices
        slices = tf.map_fn(gather_slice, slice_indices, fn_output_signature=tf.float64)  # [num_updates, 1, 5]

        # Apply activation to each slice
        def apply_activation(slice_5: tf.Tensor) -> tf.Tensor:
            return self._spatial_edge_activation(slice_5)  # returns [1,1]

        updated_features = tf.map_fn(apply_activation, slices, fn_output_signature=tf.float64)  # [num_updates, 1, 1]

        # Flatten to [num_updates]
        updated_features_flat = tf.reshape(updated_features, [-1])

        # Build scatter indices for tensor_scatter_nd_update
        scatter_indices = tf.stack([tf.zeros_like(indices), indices], axis=1)

        # Update only the second features
        outputs = tf.tensor_scatter_nd_update(inputs, scatter_indices, updated_features_flat)
        return outputs
