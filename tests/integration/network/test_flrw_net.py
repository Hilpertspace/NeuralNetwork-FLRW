import pytest
import tensorflow as tf
import numpy as np

from FLRW_Net.network.network import NeuralNetwork


@pytest.fixture(scope="module")
def setup_forward_feed_test() -> dict:
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


@pytest.fixture(scope="module")
def setup_training_test() -> dict:
    inputs = {
        "5-cell": {
            "one": tf.constant([[1, 2/5-3/8, 2]], dtype=tf.float64),
            "two": tf.constant([[1, 2/5-3/8, 1.5, 2/5-3/8, 2]], dtype=tf.float64),
            "three": tf.constant([[1, 2/5-3/8, 1.333, 2/5-3/8, 1.667, 2/5-3/8, 2]], dtype=tf.float64),
            "four": tf.constant([[1, 2/5-3/8, 1.25, 2/5-3/8, 1.5, 2/5-3/8, 1.75, 2/5-3/8, 2]], dtype=tf.float64),
        },
        "16-cell": {
            "one": tf.constant([[1, 1/2-3/8, 2]], dtype=tf.float64),
            "two": tf.constant([[1, 1/2-3/8, 1.5, 1/2-3/8, 2]], dtype=tf.float64),
            "three": tf.constant([[1, 1/2-3/8, 1.333, 1/2-3/8, 1.667, 1/2-3/8, 2]], dtype=tf.float64),
            "four": tf.constant([[1, 1/2-3/8, 1.25, 1/2-3/8, 1.5, 1/2-3/8, 1.75, 1/2-3/8, 2]], dtype=tf.float64),
        },
        "600-cell": {
            "one": tf.constant([[1, (3 + np.sqrt(5)) / 2, 2]], dtype=tf.float64),
            "two": tf.constant([[1, (3 + np.sqrt(5)) / 2, 1.5, (3 + np.sqrt(5)) / 2, 2]], dtype=tf.float64),
            "three": tf.constant([[1, (3 + np.sqrt(5)) / 2, 1.333, (3 + np.sqrt(5)) / 2, 1.667, (3 + np.sqrt(5)) / 2, 2]], dtype=tf.float64),
            "four": tf.constant([[1, (3 + np.sqrt(5)) / 2, 1.25, (3 + np.sqrt(5)) / 2, 1.5, (3 + np.sqrt(5)) / 2, 1.75, (3 + np.sqrt(5)) / 2, 2]], dtype=tf.float64),
        },
    }

    outputs = {
        "5-cell": {
            "one": tf.constant([[1, 0.6324621202151465, 2]], dtype=tf.float64),
            "two": tf.constant([[1, 0.5929270612806133, 1.937499999997557, 0.03952847075371364, 2]], dtype=tf.float64),
            "three": tf.constant([[1, 0.3952252990680342, 1.6249060669876316, 0.210875640151580883, 1.9583297299497069, 0.0263545928146881, 2]], dtype=tf.float64),
            "four": tf.constant([[1, 2]], dtype=tf.float64),
        },
        "16-cell": {
            "one": tf.constant([[1, 0.7071313364895369, 2]], dtype=tf.float64),
            "two": tf.constant([[1, 0.5303300858873097, 1.7499999999942055, 0.1767766953016012, 2]], dtype=tf.float64),
            "three": tf.constant([[1, 0.353527681421034, 1.499963641738813, 0.235722919933356, 1.8333261920694532, 0.11785617983447622, 2]], dtype=tf.float64),
            "four": tf.constant([[1, 2]], dtype=tf.float64),
        },
        "600-cell": {
            "one": tf.constant([[1, 1.6194292719298682, 2]], dtype=tf.float64),
            "two": tf.constant([[1, 0.2317627457600291, 1.1432372542000078, 1.386271243125768, 2]], dtype=tf.float64),
            "three": tf.constant([[]], dtype=tf.float64),
            "four": tf.constant([[]], dtype=tf.float64),
        },
    }

    return {

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
def test_forward_feeding(number_of_timesteps: int, string_name: str, setup_forward_feed_test: dict) -> None:
    tf.keras.backend.set_floatx("float64")
    flrw_net = NeuralNetwork(
        number_of_timesteps=number_of_timesteps,
        triangulation=setup_forward_feed_test["triangulation"],
        cosmological_constant=setup_forward_feed_test["cosmological_constant"]
    )

    tf.debugging.assert_equal(
        flrw_net(setup_forward_feed_test["inputs"][string_name]),
        setup_forward_feed_test["outputs"][string_name],
    )


def test_training() -> None:
