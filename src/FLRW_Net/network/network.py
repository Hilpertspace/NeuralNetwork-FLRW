"""Minimum example for a generalized model."""

from typing import TYPE_CHECKING

import tensorflow as tf
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]

from FLRW_Net.layers.assembly import Assembly
from FLRW_Net.layers.matmul import Matmul
from FLRW_Net.layers.relu import PartialReLU
from FLRW_Net.layers.spatial_edge_activation import SpatialEdgeActivation
from FLRW_Net.layers.strut_activation import StrutActivation
from FLRW_Net.utils.losses import spatial_edge_losses, strut_losses
from FLRW_Net.utils.utils import set_model_params

if TYPE_CHECKING:
    from FLRW_Net.utils.utils import Model


class NeuralNetwork(tf.keras.Model):
    """Minimum example for a generalized model."""

    def __init__(self, number_of_timesteps: int, triangulation: str, cosmological_constant: float) -> None:
        """Initialize the model."""
        super().__init__()
        self.matmul = Matmul(number_of_timesteps)
        self.relu_layer = PartialReLU()
        self.strut_activation_layer = StrutActivation()
        self.spatial_edge_activation = SpatialEdgeActivation()
        self.assembly = Assembly()

        self.relu_layer.trainable = False
        self.strut_activation_layer.trainable = False
        self.spatial_edge_activation.trainable = False
        self.assembly.trainable = False

        self.number_of_timesteps = tf.constant(number_of_timesteps, dtype=tf.int32)
        self.model_params: Model = set_model_params(triangulation, cosmological_constant)
        self.loss_threshold = 1e-5

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Call the model."""
        outputs_1 = self.matmul(inputs)
        outputs_2 = self.relu_layer(outputs_1)
        outputs_3 = self.strut_activation_layer(outputs_2)

        # Compute the spatial edge activation with the values from output_2 and the struts from outputs_3
        outputs_4 = tf.cond(
            self.number_of_timesteps >= 2,  # noqa: PLR2004
            lambda: self.spatial_edge_activation(outputs_3),
            lambda: outputs_3
        )

        # Use the outputs_2 scaled_as and the updated spatial edges from outputs_4 to update the strut values
        outputs_5 = self.assembly(outputs_2, outputs_4)
        outputs_6 = self.strut_activation_layer(outputs_5)

        return outputs_6

    @tf.function
    def _custom_train_step(self, inputs: tf.Tensor) -> tf.Tensor:
        """Define a single step of training."""
        with tf.GradientTape(persistent=True) as tape:
            prediction = self(inputs, training=True)

            loss_struts = strut_losses(prediction, self.model_params)
            loss_spatial_edges = spatial_edge_losses(prediction, self.model_params)
            combined = tf.concat([loss_struts, loss_spatial_edges], axis=1)
            loss = tf.reduce_mean(combined, axis=1)

        # Tell tensorflows automatic gradient computation to compute the gradients
        # of the loss with respect to the trainable variables of the network.
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply the gradients to update the weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def training(self, inputs: tf.Tensor, epochs: int) -> tuple[tf.Tensor, list[tf.Tensor]]:
        """Training logic of the neural network.

        The train_step function is called iteratively until the specified number of epochs is reached.
        """
        minimum_loss = tf.Variable(tf.float32.max, trainable=False)
        loss_history = []
        best_weights = [tf.Variable(w, trainable=False) for w in self.trainable_variables]

        for _ in tqdm(range(epochs), desc="Training", unit="step", ncols=70):
            loss = self._custom_train_step(inputs)
            loss_value = float(loss)
            loss_history.append(loss_value)

            # Update best weights
            if loss_value < float(minimum_loss):
                minimum_loss.assign(loss_value)
                for bw, w in zip(best_weights, self.trainable_variables):
                    bw.assign(w)

            # Early stopping
            if loss_value < self.loss_threshold:
                print(f"\n--- Loss below threshold ({self.loss_threshold}). Stopping. ---")
                break

        # Restore best weights
        for w, bw in zip(self.trainable_variables, best_weights):
            w.assign(bw)

        return float(minimum_loss), loss_history
