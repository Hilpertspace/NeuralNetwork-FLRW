import pytest
import tensorflow as tf

from FLRW_Net.network.network import NeuralNetwork


@pytest.fixture(scope="module")
def setup_network_test() -> dict:
    inputs = {
        "one": tf.constant([[1, 0.67, 2]], dtype=tf.float64),
        "two": tf.constant([[1, 0.67, 2, 0.68, 3]], dtype=tf.float64),
        "three": tf.constant([[1, 0.67, 2, 0.68, 3, 0.69, 4]], dtype=tf.float64),
        "four": tf.constant([[1, 0.67, 2, 0.68, 3, 0.69, 4, 0.7, 5]], dtype=tf.float64),
    }

    outputs = {
        "one": tf.constant([[1, 1.0222524150130485, 2]], dtype=tf.float64),
        "two": tf.constant([[1, 0.7301968541896805, 1.7143019116079661, 1.3205815579100626, 3]], dtype=tf.float64),
        "three": tf.constant([[1, 0.730196854189684, 1.7143019116079696, 1.0202100885913643, 2.7075629130231627, 1.3337800453372162, 4]], dtype=tf.float64),
        "four": tf.constant([[1, 0.730196854189684, 1.7143019116079696, 1.0202100885913674, 2.7075629130231658, 1.0251638043738571, 3.7009498853586638, 1.3468838258646922, 5]], dtype=tf.float64),
    }

    cosmological_constant = 1e-3
    triangulation = "5-cell"

    return {
        "inputs": inputs,
        "outputs": outputs,
        "cosmological_constant": cosmological_constant,
        "triangulation": triangulation,
    }


@pytest.mark.parametrize(
    ("number_of_timesteps", "string_name"),
    [
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
    ]
)
def test_forward_feeding(number_of_timesteps: int, string_name: str, setup_network_test: dict) -> None:
    tf.keras.backend.set_floatx("float64")
    flrw_net = NeuralNetwork(
        number_of_timesteps=number_of_timesteps,
        triangulation=setup_network_test["triangulation"],
        cosmological_constant=setup_network_test["cosmological_constant"]
    )

    tf.debugging.assert_equal(
        flrw_net(setup_network_test["inputs"][string_name]),
        setup_network_test["outputs"][string_name],
    )
