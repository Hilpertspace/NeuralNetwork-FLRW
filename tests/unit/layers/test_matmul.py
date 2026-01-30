import pytest
import tensorflow as tf

from FLRW_Net.layers.matmul import SingleSliceMatmul


@pytest.fixture(scope="module")
def setup_matmul_test() -> dict:
    inputs = tf.constant([[1, -0.5, -2, 0.5, -3]], dtype=tf.float64)

    expected_output_1 = tf.constant([[1]], dtype=tf.float64)
    expected_output_3 = tf.constant([[-0.5]], dtype=tf.float64)
    expected_output_5 = tf.constant([[-2]], dtype=tf.float64)

    return {
        "inputs": inputs,
        "expected_output_1": expected_output_1,
        "expected_output_3": expected_output_3,
        "expected_output_5": expected_output_5,
    }

def test_matmul_layer(setup_matmul_test: dict) -> None:
    tf.keras.backend.set_floatx("float64")
    inputs = setup_matmul_test["inputs"]
    expected_output_1 = setup_matmul_test["expected_output_1"]
    expected_output_3 = setup_matmul_test["expected_output_3"]
    expected_output_5 = setup_matmul_test["expected_output_5"]

    model_1 = SingleSliceMatmul(0, 1)
    output_1 = model_1(inputs)

    model_3 = SingleSliceMatmul(0, 3)
    output_3 = model_3(inputs)

    model_5 = SingleSliceMatmul(0, 5)
    output_5 = model_5(inputs)

    tf.debugging.assert_equal(output_1, expected_output_1)
    tf.debugging.assert_equal(output_3, expected_output_3)
    tf.debugging.assert_equal(output_5, expected_output_5)
