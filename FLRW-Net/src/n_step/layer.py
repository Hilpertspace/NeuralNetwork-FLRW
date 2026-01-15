"""Implements the parent of all hidden layers."""

from abc import abstractmethod

import tensorflow as tf


class HiddenLayer(tf.keras.layers.Layer):
    """Implements the parent of all hidden layers."""

    def __init__(self, number_of_timesteps: int) -> None:
        """Initialize the hidden layer.

        Args:
            number_of_timesteps (int): Number of timesteps of FLRW-Net.

        """
        self._initialize_weights(number_of_timesteps)
        self._initialize_biases(number_of_timesteps)

    @abstractmethod
    def _initialize_weights(self, number_of_timesteps: int) -> None:
        """Initialize the weights for the hidden layer."""

    @abstractmethod
    def _initialize_biases(self, number_of_timesteps: int) -> None:
        """Initialize the biases for the hidden layer."""

    @staticmethod
    def strut_activation(element_1: float, element_2: float, element_3: float) -> tf.Tensor:
        """Activation function for the strut-neurons."""
        output = tf.math.sqrt(element_2 + tf.constant(3 / 8, dtype=tf.float64))
        output *= tf.math.abs(element_1 - element_3)

        return output

    @staticmethod
    def spatial_edge_activation(l1: float, m1: float, l2: float, m2: float, l3: float) -> tf.Tensor:
        """Activation function for the spatial-edge neurons."""
        part_1 = l1 + (l3 - l1) * tf.constant(3 / 8, dtype=tf.float64) * tf.math.square(l1 - l2) / tf.math.square(m1)
        part_2 = l1 + (l3 - l1) * tf.constant(3 / 8, dtype=tf.float64) * tf.math.square(l2 - l3) / tf.math.square(m2)
        output = (part_1 + part_2) / tf.constant(2, dtype=tf.float64)

        return output

    @abstractmethod
    def _forward_feed(self, inputs) -> list[tf.Tensor]:
        """Forward-feed when the hidden layer is used."""
    
    @abstractmethod
    @staticmethod
    def _relu(inputs: list[tf.Tensor]) -> list[tf.Variable]:
        """Apply the activation function: here 'ReLU'."""

    @abstractmethod
    @staticmethod
    def _identity(inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        """Create deepcopies of the inputs."""

    @abstractmethod
    @staticmethod
    def _activate_struts(l1: float, m1: float, l2: float) -> list[tf.Tensor]
        """Apply the custom activation function ensure that
        # m1 > sqrt(3/8*(l1-l2)^2) and to compute the strut length.
        """

    @abstractmethod
    @staticmethod
    def _activate_spatial_edges() -> list[tf.Tensor]:
        # Apply the spatial-edge activation function
    
    @abstractmethod
    @staticmethod
    def _shape_output() -> tf.Tensor:
        """Shape the output correctly."""

    @abstractmethod
    def call(self, inputs, *args, **kwargs) -> tf.Tensor:
        """Use the hidden layer."""
        outputs = self._forward_feed(inputs)

        tmp = self._relu(outputs)

        scaled_a = self._identity(tmp)

        tmp_struts = self._activate_struts(outpus)

        # Apply the spatial-edge activation function
        output_3 = self._activate_spatial_edges(tmp_struts, outpus)

        tmp_struts2 = self._activate_struts(scaled_a, outputs)

        return self._shape_output(output)
