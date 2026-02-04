import pytest
import tensorflow as tf

from FLRW_Net.utils.losses import spatial_edge_losses, strut_losses
from tests.unit.utils.test_eoms import Model, model_params  # noqa: F401


# Test Tensor being of correct format for different dimensions
# Test values for two cases being of the correct format
@pytest.fixture(scope="module")
def inputs() -> dict[str, tf.Tensor]:
    one_step = tf.constant([[1, 0.66, 2]], dtype=tf.float64)
    two_step = tf.constant([[1, 0.66, 2, 0.67, 3]], dtype=tf.float64)
    three_step = tf.constant([[1, 0.66, 2, 0.67, 3, 0.68, 4]], dtype=tf.float64)
    four_step = tf.constant([[1, 0.66, 2, 0.67, 3, 0.68, 4, 0.69, 5]], dtype=tf.float64)
    five_step = tf.constant([[1, 0.66, 2, 0.67, 3, 0.68, 4, 0.69, 5, 0.7, 6]], dtype=tf.float64)
    return {
        "one_step": one_step,
        "two_step": two_step,
        "three_step": three_step,
        "four_step": four_step,
        "five_step": five_step,
    }

@pytest.fixture
def outputs() -> dict[str, dict[str, tf.Tensor]]:
    spatial_edge_losses = {
        "one_step": tf.constant([[]], dtype=tf.float64),
        "two_step": tf.constant([[1593163.2874693281]], dtype=tf.float64),
        "three_step": tf.constant([[1593163.2874693281, 893130.7750761096]], dtype=tf.float64),
        "four_step": tf.constant([[1593163.2874693281, 893130.7750761096, 515742.2686814965]], dtype=tf.float64),
        "five_step": tf.constant([[1593163.2874693281, 893130.7750761096, 515742.2686814965, 302955.0819667162]], dtype=tf.float64),
    }
    strut_losses = {
        "one_step": tf.constant([[2176.6251730934673]], dtype=tf.float64),
        "two_step": tf.constant([[2176.6251730934673, 2844.814186933328]], dtype=tf.float64),
        "three_step": tf.constant([[2176.6251730934673, 2844.814186933328, 2325.1495815007697]], dtype=tf.float64),
        "four_step": tf.constant([[2176.6251730934673, 2844.814186933328, 2325.1495815007697, 1225.2077493372142]], dtype=tf.float64),
        "five_step": tf.constant([[2176.6251730934673, 2844.814186933328, 2325.1495815007697, 1225.2077493372142, 257.91447697415725]], dtype=tf.float64),
    }

    return {
        "strut_losses": strut_losses,
        "spatial_edge_losses": spatial_edge_losses,
    }

@pytest.mark.parametrize(
    ("case", "number_of_timesteps"),
    [
        ("one_step", 1),
        ("two_step", 2),
        ("three_step", 3),
        ("four_step", 4),
        ("five_step", 5),
    ]
)
def test_strut_losses(
    case: str,
    number_of_timesteps: int,
    inputs: dict[str, tf.Tensor],
    outputs: dict[str, dict[str, tf.Tensor]],
    model_params: dict[str, Model],  # noqa: F811
) -> None:
    function_name = "strut_losses"
    output = strut_losses(prediction=inputs[case], model_params=model_params["16-cell"])
    assert tf.shape(output)[1] == number_of_timesteps

    if case != "one_step":
        tf.debugging.assert_near(
        output,
        outputs[function_name][case],
        rtol=1e-6,
        atol=1e-8,
    )

@pytest.mark.parametrize(
    ("case", "number_of_spatial_edges"),
    [
        ("one_step", 0),
        ("two_step", 1),
        ("three_step", 2),
        ("four_step", 3),
        ("five_step", 4),
    ]
)
def test_spatial_edge_losses(
    case: str,
    number_of_spatial_edges: int,
    inputs: dict[str, tf.Tensor],
    outputs: dict[str, dict[str, tf.Tensor]],
    model_params: dict[str, Model],  # noqa: F811
) -> None:
    function_name = "spatial_edge_losses"
    output = spatial_edge_losses(inputs[case], model_params["600-cell"])
    assert tf.shape(output)[1].numpy() == number_of_spatial_edges

    if case != "one_step":
        tf.debugging.assert_equal(output, outputs[function_name][case])
