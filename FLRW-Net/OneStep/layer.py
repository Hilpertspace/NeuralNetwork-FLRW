"""This module implements the single hidden layer of FLRW-Net."""

import tensorflow as tf

class HiddenLayer(tf.keras.layers.Layer):
    """
    Hidden layer of the neural network.

    Inputs:
    The two boundary edges and a parameter related to the strut: edge1, strut_param, edge2

    Returns:
    The solution for the struts given the boundary data: edge1, strut, edge2
    """

    def __init__(self):
        """
        Initialize the hidden layer.

        Parameters:
        Three trainable weights in weights_2 connecting the value of the strut to the neighbouring
        edges and strut_param. Five non-trainable parameters corresponding to the two weights for
        the boundary edges and to three biases.
        """
        super().__init__()
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        """Initialize the weights and biases."""
        # Define the weights for the three neurons respectively ...
        weights_1 = tf.constant([[1.]], dtype=tf.float64)
        weights_2 = tf.constant([[0.], [1.], [0.]], dtype=tf.float64)
        weights_3 = tf.constant([[1.]], dtype=tf.float64)

        # ... and the biases
        biases = tf.constant([0., 0., 0.], dtype=tf.float64)

        # Set the weights for the hidden layer:
        # the weights for the boundary spatial edges and the biases are set to be non-trainable
        # whereas the weights for the strut can be learned
        self.weights_1 = tf.Variable(initial_value=weights_1, trainable=False, dtype=tf.float64)
        self.weights_2 = tf.Variable(initial_value=weights_2, trainable=True, dtype=tf.float64)
        self.weights_3 = tf.Variable(initial_value=weights_3, trainable=False, dtype=tf.float64)
        self.biases = tf.Variable(initial_value=biases, trainable=False, dtype=tf.float64)

    def set_custom_weights(self, weights_2):
        """Set the trainable weights to the values of a pre-trained model."""
        self.weights_2.assign(weights_2)

    def strut_activation(self, element_1, element_2, element_3):
        """Define an activation function for the strut-neurons."""
        output = tf.math.sqrt(element_2 + tf.constant(3/8,dtype=tf.float64))
        output *= tf.math.abs(element_1 - element_3)

        return output

    def call(self, inputs, *args, **kwargs):
        """Forward-feed when the hidden layer is used."""

        # Get the individual parameters of the model, here l1 and l2
        l1 = inputs[:, 0:1]
        l2 = inputs[:, 2:3]

        # Do the forward-feed
        output_1 = tf.matmul(l1, self.weights_1) + self.biases[0]
        output_2 = tf.matmul(inputs, self.weights_2) + self.biases[1]
        output_3 = tf.matmul(l2, self.weights_3) + self.biases[2]

        # Apply the activation function: here 'ReLU'
        output_2 = tf.nn.relu(output_2) + 10 ** -14

        # Apply the custom activation function ensure that
        # m1 > sqrt(3/8*(l1-l2)^2) and to compute the strut length
        output_2 = self.strut_activation(output_1, output_2, output_3)

        # Shape the output correctly
        output = tf.concat([output_1, output_2, output_3], axis=1)

        return output
