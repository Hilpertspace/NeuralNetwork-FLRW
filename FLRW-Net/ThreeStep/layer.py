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
        edges and strut_param. Five non-trainable parameters corresponding to the weights for the
        boundary edges and to three biases.
        """
        super().__init__()

        # Define the weights for the three neurons respectively ...
        weights_1 = tf.constant([[1]], dtype=tf.float64)
        weights_2 = tf.constant([[0], [1], [0]], dtype=tf.float64)
        weights_3 = tf.constant([[0], [0], [1], [0], [0]], dtype=tf.float64)
        weights_4 = tf.constant([[0], [1], [0]], dtype=tf.float64)
        weights_5 = tf.constant([[0], [0], [1], [0], [0]], dtype=tf.float64)
        weights_6 = tf.constant([[0], [1], [0]], dtype=tf.float64)
        weights_7 = tf.constant([[1]], dtype=tf.float64)

        # ... and the biases
        biases = tf.constant([0, 0, 0, 0, 0, 0, 0], dtype=tf.float64)

        # Set the weights for the hidden layer:
        # the weights for the boundary spatial edges and the biases are set to be non-trainable
        # whereas the weights for the strut can be learned
        self.weights_1 = tf.Variable(initial_value=weights_1, trainable=False, dtype=tf.float64)
        self.weights_2 = tf.Variable(initial_value=weights_2, trainable=True, dtype=tf.float64)
        self.weights_3 = tf.Variable(initial_value=weights_3, trainable=True, dtype=tf.float64)
        self.weights_4 = tf.Variable(initial_value=weights_4, trainable=True, dtype=tf.float64)
        self.weights_5 = tf.Variable(initial_value=weights_5, trainable=True, dtype=tf.float64)
        self.weights_6 = tf.Variable(initial_value=weights_6, trainable=True, dtype=tf.float64)
        self.weights_7 = tf.Variable(initial_value=weights_7, trainable=False, dtype=tf.float64)
        self.biases = tf.Variable(initial_value=biases, trainable=False, dtype=tf.float64)

    def set_custom_weights(self, weights):
        """Set the trainable weights to the values of a pre-trained model."""
        weights_2 = tf.cast(weights[0], dtype=tf.float64)
        weights_3 = tf.cast(weights[1], dtype=tf.float64)
        weights_4 = tf.cast(weights[2], dtype=tf.float64)
        weights_5 = tf.cast(weights[3], dtype=tf.float64)
        weights_6 = tf.cast(weights[4], dtype=tf.float64)
        self.weights_2.assign(weights_2)
        self.weights_3.assign(weights_3)
        self.weights_4.assign(weights_4)
        self.weights_5.assign(weights_5)
        self.weights_6.assign(weights_6)

    def strut_activation(self, element_1, element_2, element_3):
        """Define an activation function for the strut-neurons."""
        output = tf.math.sqrt(element_2 + tf.constant(3/8,dtype=tf.float64))
        output *= tf.math.abs(element_1 - element_3)

        return output
    
    def spatial_edge_activation(self, l1, m1, l2, m2, l3):
        """Define an activation function for the spatial-edge neurons."""
        part_1 = l1 + (l3-l1) * tf.constant(3/8,dtype=tf.float64) * tf.math.square(l1-l2) / tf.math.square(m1)
        part_2 = l1 + (l3-l1) * tf.constant(3/8,dtype=tf.float64) * tf.math.square(l2-l3) / tf.math.square(m2)
        output = (part_1 + part_2) / tf.constant(2, dtype=tf.float64)

        return output

    def call(self, inputs, *args, **kwargs):
        """Forward-feed when the hidden layer is used."""

        # Get the individual parameters of the model, here l1 and l3
        l1 = inputs[:, 0:1]
        l4 = inputs[:, 6:7]

        # Do the forward-feed
        output_1 = tf.matmul(l1, self.weights_1) + self.biases[0]
        output_2 = tf.matmul(inputs[:,0:3], self.weights_2) + self.biases[1]
        output_3 = tf.matmul(inputs[:,0:5], self.weights_3) + self.biases[2]
        output_4 = tf.matmul(inputs[:,2:5], self.weights_4) + self.biases[3]
        output_5 = tf.matmul(inputs[:,2:7], self.weights_5) + self.biases[4]
        output_6 = tf.matmul(inputs[:,4:7], self.weights_6) + self.biases[5]
        output_7 = tf.matmul(l4, self.weights_7) + self.biases[6]

        # Apply the activation function: here 'ReLU'
        output_2 = tf.nn.relu(output_2) + 10 ** -14
        output_3 = tf.nn.relu(output_3) + 10 ** -14
        output_4 = tf.nn.relu(output_4) + 10 ** -14
        output_5 = tf.nn.relu(output_5) + 10 ** -14
        output_6 = tf.nn.relu(output_6) + 10 ** -14

        scaled_a1 = tf.identity(output_2)
        scaled_a2 = tf.identity(output_4)
        scaled_a3 = tf.identity(output_6)

        # Apply the custom activation function ensure that
        # m1 > sqrt(3/8*(l1-l2)^2) and to compute the strut length
        tmp_strut1 = self.strut_activation(output_1, output_2, output_3)
        tmp_strut2 = self.strut_activation(output_3, output_4, output_5)
        tmp_strut3 = self.strut_activation(output_5, output_6, output_7)

        tmp_edge2 = tf.identity(output_3)
        tmp_edge3 = tf.identity(output_5)

        # Apply the spatial-edge activation function
        output_3 = self.spatial_edge_activation(output_1, tmp_strut1, tmp_edge2, tmp_strut2, tmp_edge3)
        output_5 = self.spatial_edge_activation(tmp_edge2, tmp_strut2, tmp_edge3, tmp_strut3, output_7)

        # Ensure that
        # m1 > sqrt(3/8*(l1-l2)^2) and to compute the strut length
        output_2 = self.strut_activation(output_1, scaled_a1, output_3)
        output_4 = self.strut_activation(output_3, scaled_a2, output_5)
        output_6 = self.strut_activation(output_5, scaled_a3, output_7)

        # Shape the output correctly
        output = tf.concat([output_1, output_2, output_3, output_4, output_5, output_6, output_7], axis=1)

        return output
