"""Layer for matrix multiplication."""
import tensorflow as tf


class Matmul(tf.keras.layers.Layer):
    """Implements output_i = inputs[:, start:end] @ weight + bias."""

    dense: tf.keras.layers.Dense

    def __init__(self, start: int, end: int) -> None:
        super().__init__()
        self._start = start
        self._end = end
        self._dim = self._end - self._start
        self._output_dim = 1

        self._validate_dim(self._dim)
        self._set_default_weights()
    
    @staticmethod
    def _validate_dim(dim: int) -> None:
        """Validate the given dimension of the object.""" 
        if dim not in {3, 5}:
            msg = f"The tensor that you have specified is of wrong dimension: got {dim}, expected either 3 or 5."
            raise ValueError(msg)
    
    def _set_default_weights(self) -> None:
        """Set the default weights."""
        if self._dim == 3:
            self._default_weights = tf.constant([[0.0],[1.0],[0.0]], dtype=tf.float64)
        elif self._dim == 5:
            self._default_weights = tf.constant([[0.0],[0.0],[1.0],[0.0],[0.0]], dtype=tf.float64)            

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build a fully connected layer."""
        self.dense = tf.keras.layers.Dense(
            units=self._output_dim,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.constant_initializer(self._default_weights), # type: ignore
            bias_initializer="zeros",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Actually perform the matrix multiplication."""
        indices = tf.range(self._start, self._end)
        sliced = tf.gather(inputs, indices, axis=1)
        return self.dense(sliced)
