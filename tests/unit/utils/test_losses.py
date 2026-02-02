import pytest
import tensorflow as tf

from FLRW_Net.utils.losses import sliding_slice_losses, spatial_edge_losses, strut_losses
from tests.unit.utils.test_eoms import Model, model_params, eom_spatial_edges, eom_struts

from unittest.mock import MagicMock
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

def dummy_fn(inputs: tf.Tensor, model_params: Model) -> tf.Tensor:
    return inputs

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
def test_sliding_slice_losses_3_features(case: str, number_of_timesteps: int, inputs: dict[str, tf.Tensor], model_params: dict[str, Model]) -> None:
    output = sliding_slice_losses(
        prediction=inputs[case],
        model_params=model_params["5-cell"],
        slice_length=3,
        step=2,
        start=0,
        stop=number_of_timesteps,
        loss_fn=eom_struts,
    )

    assert tf.shape(output)[1] == number_of_timesteps

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
def test_sliding_slice_losses_5_features(case: str, number_of_spatial_edges: int, inputs: dict[str, tf.Tensor], model_params: dict[str, Model]) -> None:
    if case == "one_step":
        # This should not occur and be jumped over
        pass
    output = sliding_slice_losses(
        prediction=inputs[case],
        model_params=model_params["16-cell"],
        slice_length=5,
        step=2,
        start=1,
        stop=1+number_of_spatial_edges,
        loss_fn=eom_spatial_edges,
    )

    assert tf.shape(output)[1].numpy() == number_of_spatial_edges

def test_spatial_edge_losses() -> None:
    assert False


def test_strut_losses() -> None:
    assert False
