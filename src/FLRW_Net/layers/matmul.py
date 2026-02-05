"""Layer for matrix multiplication."""
import tensorflow as tf

from FLRW_Net.utils.utils import set_slice_specs, sort_slice_specs


class SingleSliceMatmul(tf.keras.layers.Layer):
    """Implements output = inputs[:, start:end] @ weight + bias for a single slice."""

    dense: tf.keras.layers.Dense

    def __init__(self, start: int, end: int) -> None:
        """Initialize the layer."""
        super().__init__()
        self._start = start
        self._end = end
        self._dim = self._end - self._start

        self._boundary_edge_input_dim = 1
        self._strut_input_dim = 3
        self._spatial_edge_input_dim = 5
        self._output_dim = 1

        self._use_identity = False

        self._validate_dim(self._dim)

    @staticmethod
    def _validate_dim(dim: int) -> None:
        """Validate the given dimension of the object."""
        if dim not in {1, 3, 5}:
            msg = f"The tensor that you have specified is of wrong dimension: got {dim}, expected either 1, 3 or 5."
            raise ValueError(msg)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build a fully connected layer."""
        if self._dim == self._boundary_edge_input_dim:
            self._use_identity = True
        else:
            if self._dim == self._strut_input_dim:
                self._default_weights = [[0.0],[1.0],[0.0]]
            elif self._dim == self._spatial_edge_input_dim:
                self._default_weights = [[0.0],[0.0],[1.0],[0.0],[0.0]]

            self.dense = tf.keras.layers.Dense(
                units=self._output_dim,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.constant_initializer(self._default_weights), # type: ignore
                bias_initializer="zeros",
            )
            super().build(input_shape)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Actually perform the matrix multiplication."""
        indices = tf.range(self._start, self._end)
        sliced = tf.gather(inputs, indices, axis=1)

        if self._use_identity:
            return sliced
        return self.dense(sliced)

class Matmul(tf.keras.layers.Layer):
    """Implements the forward feed for all slices."""

    def __init__(self, number_of_timesteps: int) -> None:
        """Initialize the layer."""
        super().__init__()
        self._slice_specs = sort_slice_specs(set_slice_specs(number_of_timesteps))
        self.slice_layers = [SingleSliceMatmul(start, end) for start, end in self._slice_specs]

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply matrix multiplication to the inputs."""
        outputs = [layer(inputs) for layer in self.slice_layers]
        return tf.concat(outputs, axis=1)
