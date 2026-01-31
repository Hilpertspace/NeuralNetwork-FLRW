import numpy as np
import tensorflow as tf

from FLRW_Net.utils.utils import Model, crop, assert_non_negative


@tf.function
def eom_struts(prediction: tf.Tensor, model_params: Model) -> tf.Tensor:
    """Compute the network's loss by evaluating the EOMm term.

    Its value should be 0 in the presence of a classical solution.
    """
    # Define the parts of the output in Regge calculus variables
    l1 = prediction[0, 0]
    m1 = prediction[0, 1]
    l2 = prediction[0, 2]

    assert_non_negative(l1, "l1")
    assert_non_negative(m1, "m1")
    assert_non_negative(l2, "l2")

    lamb = model_params.lamb
    n3 = model_params.n3
    n1 = model_params.n1
    nte = model_params.nte

    pi = tf.constant(np.pi, dtype=tf.float64)

    # Compute the value of the EOMm: the first fraction
    numerator1 = -(l1 + l2) * (tf.math.square(l1) + tf.math.square(l2)) * lamb * m1 * n3
    denominator1 = tf.constant(12, dtype=tf.float64) * tf.math.sqrt(
        tf.constant(-3, dtype=tf.float64) * tf.math.square(l1 - l2) + tf.constant(8, dtype=tf.float64) * tf.math.square(m1)
    )

    # Compute the value of the EOMm: the second fraction
    # 'crop' ensures the arguments of the arccos to be indeed in [-1, 1]
    arg = crop(
        (tf.math.square(l1 - l2) - tf.constant(2, dtype=tf.float64) * tf.math.square(m1))
        / (tf.constant(2, dtype=tf.float64) * tf.math.square(l1 - l2) - tf.constant(6, dtype=tf.float64) * tf.math.square(m1))
    )
    numerator2 = (l1 + l2) * m1 * n1 * (tf.constant(2, dtype=tf.float64) * pi - nte * tf.math.acos(arg))
    denominator2 = tf.math.sqrt(
        -tf.constant(1, dtype=tf.float64) * tf.math.square(l1 - l2) + tf.constant(4, dtype=tf.float64) * tf.math.square(m1)
    )

    # Define the total loss as: EOMm ^ 2. This ensures that the solution of the time step
    # can be found as a minimizing procedure, since loss = 0 <--> EOMm = 0 i.e.,
    # a classical solution
    loss = tf.math.square(numerator1 / denominator1 + numerator2 / denominator2)

    return loss

@tf.function
def arg1(inputs: tf.Tensor) -> tf.Tensor:
    """Argument of dihedral angle 1."""
    l1 = inputs[0, 0]
    m1 = inputs[0, 1]
    l2 = inputs[0, 2]
    return crop(
        (tf.math.square(l1 - l2) - tf.constant(2, dtype=tf.float64) * tf.math.square(m1))
        / (tf.constant(2, dtype=tf.float64) * tf.math.square(l1 - l2) - tf.constant(6, dtype=tf.float64) * tf.math.square(m1))
    )

@tf.function
def arg2(inputs: tf.Tensor) -> tf.Tensor:
    """Argument of dihedral angle 2."""
    l1 = inputs[0, 0]
    m1 = inputs[0, 1]
    l2 = inputs[0, 2]
    return crop(
        (-l1 + l2)
        / (
            tf.constant(2.0, dtype=tf.float64)
            * tf.math.sqrt(
                tf.constant(-2.0, dtype=tf.float64) * tf.math.square(l1 - l2) + tf.constant(6.0, dtype=tf.float64) * tf.math.square(m1)
            )
        )
    )

@tf.function
def eoml1(inputs: tf.Tensor, model_params: Model) -> tf.Tensor:
    """Part of the equation of motion of the spatial edge that has the number of tetrahedra n3 as a prefactor."""
    l1 = inputs[0, 0]
    m1 = inputs[0, 1]
    l2 = inputs[0, 2]

    lamb = model_params.lamb
    n3 = model_params.n3

    output = (
        1
        / 24
        * n3
        * lamb
        * (3 * tf.math.pow(l2, 3) * (-l1 + l2) - 2 * (tf.math.square(l1 + l2) + 2 * tf.math.square(l2)) * tf.math.square(m1))
        / tf.math.sqrt(-3 * tf.math.square(l1 - l2) + 8 * tf.math.square(m1))
    )
    return output

@tf.function
def eoml_acoses(inputs: tf.Tensor, model_params: Model) -> tf.Tensor:
    """Part of the equation of motion of the spatial edge that has the number of triangles n2 as a prefactor."""
    l2 = inputs[0, 2]
    n2 = model_params.n2
    pi = tf.constant(np.pi, dtype=tf.float64)

    tmp1 = crop(arg2(inputs[:, :3]))
    tmp2 = crop(arg2(tf.reverse(inputs[:, 2:], axis=[1])))
    output = tf.math.sqrt(tf.constant(3.0, dtype=tf.float64)) * l2 * n2 * (pi - tf.math.acos(tmp1) - tf.math.acos(tmp2))
    return output

@tf.function
def eoml2(inputs: tf.Tensor, model_params: Model) -> tf.Tensor:
    """Part of the equation of motion of the spatial edge that has the number of edges n1 as a prefactor."""
    l1 = inputs[0, 0]
    m1 = inputs[0, 1]
    l2 = inputs[0, 2]
    m2 = inputs[0, 3]
    l3 = inputs[0, 4]

    n1 = model_params.n1
    nte = model_params.nte

    pi = tf.constant(np.pi, dtype=tf.float64)

    output = (
        n1
        * (2 * pi - nte * tf.math.acos(crop(arg1(inputs[:, :3]))))
        * tf.math.sqrt(-tf.math.square(l2 - l3) + 4 * tf.math.square(m2))
        * (-2 * tf.math.square(m1) - l1 * l2 + tf.math.square(l2))
        / (
            2
            * tf.math.sqrt(-tf.math.square(l1 - l2) + 4 * tf.math.square(m1))
            * tf.math.sqrt(-tf.math.square(l2 - l3) + 4 * tf.math.square(m2))
        )
    )
    return output

@tf.function
def eom_spatial_edges(inputs: tf.Tensor, model_params: Model) -> tf.Tensor:
    """Final EOMl term ^ 2."""
    term1 = eoml1(inputs[:, :3], model_params)
    term2 = eoml1(tf.reverse(inputs[:, 2:], axis=[1]), model_params)
    term3 = eoml_acoses(inputs, model_params)
    term4 = eoml2(inputs, model_params)
    term5 = eoml2(tf.reverse(inputs, axis=[1]), model_params)
    return tf.math.square(term1 + term2 + term3 - term4 - term5)
