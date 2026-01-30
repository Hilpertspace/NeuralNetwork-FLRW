import pytest
import tensorflow as tf

from FLRW_Net.layers.spatial_edge_activation import SpatialEdgeActivation


@pytest.fixture(scope="module")
def setup_strut_activation_test() -> dict:
    inputs_3 = tf.constant([[1, 0.5, 2, 0.5, 3, 0.6, 4]], dtype=tf.float64)
    inputs_2 = tf.constant([[1, 0.5, 3.5, 0.5, 4]], dtype=tf.float64)

    expected_output_3 = tf.constant([[1, 0.5, 4, 0.5, 4.54166667, 0.6, 4]], dtype=tf.float64)
    expected_output_2 = tf.constant([[1, 0.5, 15.625, 0.5, 4]], dtype=tf.float64)

    return {
        "inputs_2": inputs_2,
        "inputs_3": inputs_3,
        "expected_output_3": expected_output_3,
        "expected_output_2": expected_output_2,
    }

def test_relu_layer(setup_strut_activation_test: dict) -> None:
    tf.keras.backend.set_floatx("float64")
    inputs_3 = setup_strut_activation_test["inputs_3"]
    inputs_2 = setup_strut_activation_test["inputs_2"]
    expected_output_3 = setup_strut_activation_test["expected_output_3"]
    expected_output_2 = setup_strut_activation_test["expected_output_2"]

    strut_activation_layer = SpatialEdgeActivation()
    output_3 = strut_activation_layer(inputs_3)
    output_2 = strut_activation_layer(inputs_2)

    tf.debugging.assert_near(output_3, expected_output_3, atol=1e-8, rtol=1e-8)
    tf.debugging.assert_near(output_2, expected_output_2, atol=1e-8, rtol=1e-8) # TODO: check how accurate these values are
