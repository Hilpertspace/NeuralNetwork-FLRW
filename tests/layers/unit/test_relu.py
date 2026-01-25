import pytest
import tensorflow as tf

from FLRW_Net.layers.relu import PartialReLU

@pytest.fixture(scope="module")
def setup_matmul_test() -> dict:
    inputs_3 = tf.constant([[-1, 0.5, -1, 0.5, 3, -0.5, 4]], dtype=tf.float64)
    inputs_2 = tf.constant([[1, 0.5, 2, 0.5, -3]], dtype=tf.float64)

    expected_output_3 = tf.constant([[-1, 0.5, 0, 0.5, 3, 0, 4]], shape=(1, 7), dtype=tf.float64)
    expected_output_2 = tf.constant([[1, 0.5, 2, 0.5, -3]], shape=(1, 5), dtype=tf.float64)

    return {
        "inputs_2": inputs_2,
        "inputs_3": inputs_3,
        "expected_output_3": expected_output_3,
        "expected_output_2": expected_output_2,
    }

def test_relu_layer(setup_matmul_test: dict) -> None:
    tf.keras.backend.set_floatx("float64")
    inputs_3 = setup_matmul_test["inputs_3"]
    inputs_2 = setup_matmul_test["inputs_2"]
    expected_output_3 = setup_matmul_test["expected_output_3"]
    expected_output_2 = setup_matmul_test["expected_output_2"]

    relu_layer = PartialReLU()
    output_3 = relu_layer(inputs_3)
    output_2 = relu_layer(inputs_2)

    tf.debugging.assert_equal(output_3, expected_output_3)
    tf.debugging.assert_equal(output_2, expected_output_2)
