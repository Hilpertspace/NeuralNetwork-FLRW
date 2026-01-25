import pytest
import tensorflow as tf

from FLRW_Net.layers.assembly import Assembly


@pytest.fixture(scope="module")
def setup_matmul_test() -> dict:
    outputs_2 = tf.constant([[1, 0.5, 2, 0.5, 3, 0.6, 4]], dtype=tf.float64)
    outputs_4 = tf.constant([[1, 2, 2.2, 4, 3.1, 6, 7]], dtype=tf.float64)

    expected_output = tf.constant([[1, 0.5, 2.2, 0.5, 3.1, 0.6, 4]], dtype=tf.float64)

    return {
        "outputs_2": outputs_2,
        "outputs_4": outputs_4,
        "expected_output": expected_output,
    }

def test_matmul_layer(setup_matmul_test: dict) -> None:
    tf.keras.backend.set_floatx("float64")
    outputs_2 = setup_matmul_test["outputs_2"]
    outputs_4 = setup_matmul_test["outputs_4"]
    expected_output = setup_matmul_test["expected_output"]

    assembly_layer = Assembly()
    output = assembly_layer(outputs_2=outputs_2, outputs_4=outputs_4)

    tf.debugging.assert_near(output, expected_output, atol=1e-8, rtol=1e-8)
