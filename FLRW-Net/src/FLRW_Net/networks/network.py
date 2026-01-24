"""Base Network class for FLRW-Net."""

import tkinter as tk
from tkinter import ttk

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from OneStep.layer import HiddenLayer
from abc import abstractmethod

from FLRW_Net.utils.utils import get_triangulation_params

class Network(tf.keras.Model):
    """Base Network class for FLRW-Net."""

    hidden_layer: tf.keras.layers.Layer

    def __init__(self, triangulation: str, training_params: dict[str, float]) -> None:
        """Initialize the neural network.

        Regge calculus parameters of a spatial triangulation:
        n1:   number of edges
        n2:   number of faces
        n3:   number of tetrahedra
        nte:  number of triangles per edge
        lamb: value of the cosmological constant

        Evaluation params:
        loss_array:   saves performance over iterations
        minimum_loss: stores the minimum loss occurred during training
        argmin: position of the minimum loss in loss_array

        """
        super().__init__(self)

        self.pi = tf.constant(np.pi, dtype=tf.float64)
        self.one = tf.constant(1.0, dtype=tf.float64)
        self.minus_one = tf.constant(-1.0, dtype=tf.float64)

        triangulation_params = get_triangulation_params(triangulation=triangulation)

        # Define Regge calculus parameters
        self.n1 = tf.constant(triangulation_params["n1"], dtype=tf.float64)
        self.n2 = tf.constant(triangulation_params["n2"], dtype=tf.float64)
        self.n3 = tf.constant(triangulation_params["n3"], dtype=tf.float64)
        self.nte = tf.constant(triangulation_params["nte"], dtype=tf.float64)

        self.lamb = tf.constant(training_params["cosmological_constant"], dtype=tf.float64)
        self.loss_threshold = training_params["loss_threshold"]

        self.min_weights = None

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward-feed when the model is used."""
        # Convert inputs to dtype=tf.float64
        inputs = tf.cast(inputs, dtype=tf.float64)

        # Only the hidden layer is called and its forward-feed will be executed
        return self.hidden_layer(inputs)

    def crop(self, arg):
        """Set the argument range of 'tf.acos' to [-1., 1.] since numeric computations could
        lead to values that are slightly outside this range due to computational error.
        """
        # Define the conditions
        too_small = arg < self.minus_one
        too_large = arg > self.one

        # Use tensorflow's if/then/else: (if condition, then, else(if condition2, then, else))
        output = tf.where(too_small, self.minus_one, tf.where(too_large, self.one, arg))

        return output

    def set_weights(self, weights):
        """Get trainable weights of a pre-trained model and hand it to the hidden layer."""
        weights = tf.cast(weights, dtype=tf.float64)
        self.hidden.set_custom_weights(weights)

    def get_min_weights(self):
        """Output the trainable weights of the hidden layer that produced the minimum loss."""
        return self.min_weights

    def custom_compute_loss(self, prediction):
        """Compute the network's loss by evaluating the EOMm term, which should be 0 in the
        presence of a classical solution. Thus, its value represents a natural choice for the loss.
        """
        # Define the parts of the output in Regge calculus variables
        l1 = prediction[0, 0]
        m1 = prediction[0, 1]
        l2 = prediction[0, 2]

        # Compute the value of the EOMm: the first fraction
        numerator1 = -(l1 + l2) * (tf.math.square(l1) + tf.math.square(l2)) * self.lamb * m1 * self.n3
        denominator1 = tf.constant(12, dtype=tf.float64) * tf.math.sqrt(
            tf.constant(-3, dtype=tf.float64) * tf.math.square(l1 - l2) + tf.constant(8, dtype=tf.float64) * tf.math.square(m1)
        )

        # Compute the value of the EOMm: the second fraction
        # 'self.crop' ensures the arguments of the arccos to be indeed in [-1, 1]
        arg = self.crop(
            (tf.math.square(l1 - l2) - tf.constant(2, dtype=tf.float64) * tf.math.square(m1))
            / (tf.constant(2, dtype=tf.float64) * tf.math.square(l1 - l2) - tf.constant(6, dtype=tf.float64) * tf.math.square(m1))
        )
        numerator2 = (l1 + l2) * m1 * self.n1 * (tf.constant(2, dtype=tf.float64) * self.pi - self.nte * tf.math.acos(arg))
        denominator2 = tf.math.sqrt(
            -tf.constant(1, dtype=tf.float64) * tf.math.square(l1 - l2) + tf.constant(4, dtype=tf.float64) * tf.math.square(m1)
        )

        # Define the total loss as: EOMm ^ 2. This ensures that the solution of the time step
        # can be found as a minimizing procedure, since loss = 0 <--> EOMm = 0 i.e.,
        # a classical solution
        loss = tf.math.square(numerator1 / denominator1 + numerator2 / denominator2)

        return loss

    def abort(self):
        """Abort the training if the according button is clicked in the training window."""
        self.abort_training = True

    def training(self, inputs, epochs):
        """Training logic of the neural network. The train_step function is called iteratively until
        the specified number of epochs is reached.
        """
        # Convert inputs to dtype=tf.float64
        inputs = tf.cast(inputs, dtype=tf.float64)

        loss_array = []

        # Create the progress bar
        training_progress = ttk.Progressbar(self.training_window, orient=tk.HORIZONTAL, length=300, mode="determinate")

        for i in tqdm(range(epochs), desc="Progress", unit="step", ncols=69):
            if self.abort_training:
                print("\rTraining aborted.")
                return None, None

            # Update the progress bar
            training_progress["value"] = i
            training_progress.update()
            training_progress.update_idletasks()

            # Save current values of the weights
            tmp_weights = tf.identity(self.hidden.trainable_weights)

            # Compute the loss and update the trainable weights
            loss_array.append(self.custom_train_step(inputs)["loss"].numpy())

            # If the last loss is lower than all before, update the minimum parameters
            argmin = np.argmin(loss_array)
            if argmin == len(loss_array) - 1:
                self.min_weights = tf.identity(tmp_weights)
                minimum_loss = loss_array[-1]

            # If the latest loss is smaller that the minimum threshold, abort training
            if loss_array[-1] < self.loss_threshold:
                print(f"\r--- Loss below specified threshold of {self.loss_threshold}.", " Training completed. ---\n")
                return minimum_loss, loss_array

        return minimum_loss, loss_array

    @tf.function
    def custom_train_step(self, inputs):
        """Define a single step of training.
        '@tf.function' speeds up training incredibly: --- DO NOT REMOVE! ---
        """
        # Use the automatic gradient computation
        with tf.GradientTape(persistent=True) as tape:
            # Perform 1 forward-feed step
            prediction = self(inputs, training=True)  # Forward pass

            loss = self.custom_compute_loss(prediction)

        # Tell tensorflows automatic gradient computation to compute the gradients
        # of the loss with respect to the trainable variables of the network:
        # here only the three weights of the strut-neuron in the hidden layer
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply the gradients to update the weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}
