"""Minimum example for a generalized model."""

import tensorflow as tf

from FLRW_Net.layers.assembly import Assembly
from FLRW_Net.layers.matmul import Matmul
from FLRW_Net.layers.relu import PartialReLU
from FLRW_Net.layers.spatial_edge_activation import SpatialEdgeActivation
from FLRW_Net.layers.strut_activation import StrutActivation


class GeneralizedModel(tf.keras.Model):
    """Minimum example for a generalized model."""

    def __init__(self, slice_specs: list[tuple[int, int]]) -> None:
        """Initialize the model."""
        super().__init__()
        self.slice_layers = [Matmul(start, end) for start, end in slice_specs]
        self.relu_layer = PartialReLU()
        self.strut_activation_layer = StrutActivation()
        self.spatial_edge_activation = SpatialEdgeActivation()
        self.assembly = Assembly()

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Call the model."""
        outputs_1 = [layer(inputs) for layer in self.slice_layers]
        outputs_1 = tf.concat(outputs_1, axis=1)

        outputs_2 = self.relu_layer(outputs_1)
        outputs_3 = self.strut_activation_layer(outputs_2)

        # Compute the spatial edge activation with the values from output_2 and the struts from outputs_3
        outputs_4 = self.spatial_edge_activation(outputs_3)

        # Use the outputs_2 scaled_as and the updated spatial edges from outputs_4 to update the strut values
        outputs_5 = self.assembly(outputs_2, outputs_4)
        outputs_6 = self.strut_activation_layer(outputs_5)

        return outputs_6
