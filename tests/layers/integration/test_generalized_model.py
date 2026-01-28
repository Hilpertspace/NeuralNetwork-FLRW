import pytest
import tensorflow as tf

from FLRW_Net.networks.test import GeneralizedModel


@pytest.fixture(scope="module")
def setup_network_test() -> dict:
    inputs_3 = tf.constant([[1, 0.5, 2, 0.5, 3, 0.6, 4]], dtype=tf.float64)
    inputs_2 = tf.constant([[1, 0.5, 3.5, 0.5, 4]], dtype=tf.float64)

    expected_output_3 = tf.constant([[1, 0.8017837257372732, 1.85714286, 0.89429723, 2.81318681, 1.17188412, 4]], dtype=tf.float64)
    expected_output_2 = tf.constant([[1, 1.20267559, 2.28571429, 1.60356745, 4]], dtype=tf.float64)

    return {
        "inputs_2": inputs_2,
        "inputs_3": inputs_3,
        "expected_output_3": expected_output_3,
        "expected_output_2": expected_output_2,
    }

def test_generalized_network(setup_network_test: dict) -> None:
    tf.keras.backend.set_floatx("float64")
    inputs_3 = setup_network_test["inputs_3"]
    inputs_2 = setup_network_test["inputs_2"]
    expected_output_3 = setup_network_test["expected_output_3"]
    expected_output_2 = setup_network_test["expected_output_2"]

    model_3 = GeneralizedModel(number_of_timesteps=3)
    output_3 = model_3(inputs_3)

    model_2 = GeneralizedModel(number_of_timesteps=2)
    output_2 = model_2(inputs_2)

    tf.debugging.assert_near(output_3, expected_output_3, atol=1e-8, rtol=1e-8)
    tf.debugging.assert_near(output_2, expected_output_2, atol=1e-8, rtol=1e-8) # TODO: Check whether the computation matches the previous accuracy  # noqa: E501
