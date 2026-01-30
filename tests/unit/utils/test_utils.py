import pytest
import tensorflow as tf

from FLRW_Net.utils.utils import crop, get_triangulation_params, set_slice_specs, Triangulation


@pytest.fixture(scope="module")
def utils_test_data() -> dict:
    slice_specs_1 = [
        (0, 1),
        (0, 3),
        (2, 3),
    ]
    slice_specs_2 = [
        (0, 1),
        (0, 3),
        (2, 5),
        (0, 5),
        (4, 5),
    ]
    slice_specs_3 = [
        (0, 1),
        (0, 3),
        (2, 5),
        (4, 7),
        (0, 5),
        (2, 7),
        (6, 7),
    ]
    slice_specs_4 = [
        (0, 1),
        (0, 3),
        (2, 5),
        (4, 7),
        (6, 9),
        (0, 5),
        (2, 7),
        (4, 9),
        (8, 9),
    ]
    return {
        "slice_specs_1": slice_specs_1,
        "slice_specs_2": slice_specs_2,
        "slice_specs_3": slice_specs_3,
        "slice_specs_4": slice_specs_4,
    }

@pytest.mark.parametrize(
    "value",
    [
        tf.constant(-2, dtype=tf.float64),
        tf.constant(-1, dtype=tf.float64),
        tf.constant(-1.0, dtype=tf.float64),
        tf.constant(0.1, dtype=tf.float64),
        tf.constant(1, dtype=tf.float64),
        tf.constant(1.0, dtype=tf.float64),
        tf.constant(2, dtype=tf.float64),
    ]
)
def test_crop(value: tf.Tensor) -> None:
    low = tf.constant(-1, dtype=tf.float64)
    high = tf.constant(1, dtype=tf.float64)

    tf.debugging.assert_greater_equal(crop(value), low)
    tf.debugging.assert_less_equal(crop(value), high)

@pytest.mark.parametrize(
    "triangulation",
    [
        "5-cell",
        "16-cell",
        "600-cell",
        "invalid",
    ]
)
def test_get_triangualtion_params(triangulation: str) -> None:
    if triangulation != "invalid":
        params = get_triangulation_params(triangulation)
        assert isinstance(params, Triangulation)
        if triangulation == "5-cell":
            model = {
                "n1": 10,
                "n2": 10,
                "n3": 5,
                "nte": 3,
            }
        elif triangulation == "16-cell":
            model = {
                "n1": 24,
                "n2": 32,
                "n3": 16,
                "nte": 4,
            }
        else:
            model = {
                "n1": 720,
                "n2": 1200,
                "n3": 600,
                "nte": 5,
            }
        assert model == params._asdict()
    else:
        with pytest.raises(
            ValueError, match=f"The given triangulation {triangulation} is invalid. Please choose from '5-cell', '16-cell', '600-cell'."
        ):
            get_triangulation_params(triangulation)

@pytest.mark.parametrize(
    ("number_of_timesteps", "slice_specs_name"),
    [
        (1, "slice_specs_1"),
        (2, "slice_specs_2"),
        (3, "slice_specs_3"),
        (4, "slice_specs_4"),
    ]
)
def test_set_slice_specs(number_of_timesteps: int, slice_specs_name: str, utils_test_data: dict[str, list[tuple]]) -> None:
    assert set_slice_specs(number_of_timesteps) == utils_test_data[slice_specs_name]
