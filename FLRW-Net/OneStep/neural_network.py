"""This module implements FLRW-Net."""

import tkinter as tk
from tkinter import ttk

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from OneStep.layer import HiddenLayer

class Network(tf.keras.Model):
    """
    Definition of FLRW-Net.
    The network takes the boundary edges and solves the EOMm for the struts, which it returns.
    """

    def __init__(self,training_window,n1=10,n3=5,nte=3,lamb=1e-2,loss_threshold=1e-26,**kwargs):
        """
        Initialize the neural network.

        UI params:
        window: training_window, flag: abort_training, button: button_abort

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
        super().__init__(self, **kwargs)
        self.initialize_params(n1, n3, nte, lamb, loss_threshold, training_window)

        # Initialize abortion button
        button_abort = tk.Button(training_window, text="Abort", background="lightgray",
                                 command=self.abort)
        button_abort.grid(row=1, column=0, columnspan=1, padx=5, pady=5)
        button_abort.config(state="normal")

        # Make the hidden layer callable as 'self.hidden'
        self.hidden = HiddenLayer()

    def initialize_params(self, n1, n3, nte, lamb, loss_threshold, training_window):
        """Initialize training constants."""
        # Define global constants for the model in 'float64' format
        self.pi = tf.constant(np.pi, dtype=tf.float64)
        self.one = tf.constant(1., dtype=tf.float64)
        self.minus_one = tf.constant(-1., dtype=tf.float64)

        # Define Regge calculus parameters
        self.n1 = tf.constant(n1, dtype=tf.float64)
        self.n3 = tf.constant(n3, dtype=tf.float64)
        self.nte = tf.constant(nte, dtype=tf.float64)
        self.lamb = tf.constant(lamb, dtype=tf.float64)

        self.min_weights = None
        self.loss_threshold = loss_threshold

        self.training_window = training_window
        self.abort_training = False

    def call(self, inputs):
        """Forward-feed when the model is used."""
        # Convert inputs to dtype=tf.float64
        inputs = tf.cast(inputs, dtype=tf.float64)

        # Only the hidden layer is called and its forward-feed will be executed
        return self.hidden(inputs)

    def crop(self, arg):
        """
        Set the argument range of 'tf.acos' to [-1., 1.] since numeric computations could
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
        """
        Compute the network's loss by evaluating the EOMm term, which should be 0 in the
        presence of a classical solution. Thus, its value represents a natural choice for the loss.
        """
        # Define the parts of the output in Regge calculus variables
        l1 = prediction[0,0]
        m1 = prediction[0,1]
        l2 = prediction[0,2]

        # Compute the value of the EOMm: the first fraction
        numerator1 = -(l1 + l2) * (tf.math.square(l1) + tf.math.square(l2)) * self.lamb*m1*self.n3
        denominator1 = tf.constant(12,dtype=tf.float64) * tf.math.sqrt(tf.constant(-3,dtype=tf.float64) * tf.math.square(l1 - l2) + tf.constant(8,dtype=tf.float64) * tf.math.square(m1))

        # Compute the value of the EOMm: the second fraction
        # 'self.crop' ensures the arguments of the arccos to be indeed in [-1, 1]
        arg = self.crop( (tf.math.square(l1 - l2) - tf.constant(2,dtype=tf.float64) * tf.math.square(m1)) / (tf.constant(2,dtype=tf.float64) * tf.math.square(l1 - l2) - tf.constant(6,dtype=tf.float64) * tf.math.square(m1)) )
        numerator2 = (l1 + l2) * m1 * self.n1 * (tf.constant(2,dtype=tf.float64) * self.pi - self.nte * tf.math.acos(arg) )
        denominator2 = tf.math.sqrt(-tf.constant(1,dtype=tf.float64)*tf.math.square(l1-l2)+tf.constant(4,dtype=tf.float64)*tf.math.square(m1))

        # Define the total loss as: EOMm ^ 2. This ensures that the solution of the time step
        # can be found as a minimizing procedure, since loss = 0 <--> EOMm = 0 i.e.,
        # a classical solution
        loss = tf.math.square(numerator1 / denominator1 + numerator2 / denominator2)

        return loss

    def abort(self):
        """Abort the training if the according button is clicked in the training window."""
        self.abort_training = True

    def training(self, inputs, epochs):
        """
        Training logic of the neural network. The train_step function is called iteratively until
        the specified number of epochs is reached.
        """
        # Convert inputs to dtype=tf.float64
        inputs = tf.cast(inputs, dtype=tf.float64)

        loss_array = []

        # Create the progress bar
        training_progress = ttk.Progressbar(self.training_window, orient=tk.HORIZONTAL, length=300,
                                            mode='determinate')

        for i in tqdm(range(epochs), desc='Progress', unit='step', ncols=69):
            if self.abort_training:
                print("\rTraining aborted.")
                return None, None

            # Update the progress bar
            training_progress['value'] = i
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
                print(f"\r--- Loss below specified threshold of {self.loss_threshold}.",
                         " Training completed. ---\n")
                return minimum_loss, loss_array

        return minimum_loss, loss_array

    @tf.function
    def custom_train_step(self, inputs):
        """
        Define a single step of training.
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
