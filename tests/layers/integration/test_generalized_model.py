import pytest
import tensorflow as tf

from FLRW_Net.networks.test import GeneralizedModel


@pytest.fixture(scope="module")
def setup_network_test() -> dict:
    slice_specs_3 = [
        (0, 1),   # l1
        (0, 3),   # first 3 inputs
        (0, 5),   # first 5 inputs
        (2, 5),   # inputs[2:5]
        (2, 7),   # inputs[2:7]
        (4, 7),   # inputs[4:7]
        (6, 7),   # l4
    ]

    slice_specs_2 = [
        (0, 1),   # l1
        (0, 3),   # first 3 inputs
        (0, 5),   # full input, adjust if necessary
        (2, 5),   # inputs[2:]
        (4, 5),   # l3
    ]

    inputs_3 = tf.constant([[-1, 0.5, -1, 0.5, 3, -0.5, 4]], dtype=tf.float64)
    inputs_2 = tf.constant([[1, 0.5, 2, 0.5, -3]], dtype=tf.float64)

    expected_output_3 = tf.constant([[-1, 0.5, 0, 0.5, 3, 0, 4]], shape=(1, 7), dtype=tf.float64)
    expected_output_2 = tf.constant([[1, 0.5, 2, 0.5, -3]], shape=(1, 5), dtype=tf.float64)

    return {
        "slice_specs_2": slice_specs_2,
        "slice_specs_3": slice_specs_3,
        "inputs_2": inputs_2,
        "inputs_3": inputs_3,
        "expected_output_3": expected_output_3,
        "expected_output_2": expected_output_2,
    }

def test_generalized_network(setup_network_test: dict) -> None:
    tf.keras.backend.set_floatx("float64")
    slice_specs_3 = setup_network_test["slice_specs_3"]
    slice_specs_2 = setup_network_test["slice_specs_2"]
    inputs_3 = setup_network_test["inputs_3"]
    inputs_2 = setup_network_test["inputs_2"]
    expected_output_3 = setup_network_test["expected_output_3"]
    expected_output_2 = setup_network_test["expected_output_2"]

    model_3 = GeneralizedModel(slice_specs_3)
    output_3 = model_3(inputs_3)

    model_2 = GeneralizedModel(slice_specs_2)
    output_2 = model_2(inputs_2)

    tf.debugging.assert_equal(output_3, expected_output_3)
    tf.debugging.assert_equal(output_2, expected_output_2)
