import pytest
import tensorflow as tf

from FLRW_Net.utils.eoms import arg1, arg2, eom_spatial_edges, eom_struts, eoml1, eoml2, eoml_acoses
from FLRW_Net.utils.utils import Model, set_model_params


@pytest.fixture(scope="module")
def inputs() -> dict:
    eom_struts = {
        "invalid_config": tf.constant([[1, 0.6, 2]], dtype=tf.float64),
        "working_edge_small_diff": tf.constant([[1, 0.67, 2]], dtype=tf.float64),
        "working_large_struts": tf.constant([[1, 10.1, 2]], dtype=tf.float64),
        "working_large_spatial_edges": tf.constant([[100, 70.1, 200]], dtype=tf.float64),
        "invalid_first_edge": tf.constant([[-1, 0.7, 2]], dtype=tf.float64),
        "invalid_strut": tf.constant([[1, -0.7, 2]], dtype=tf.float64),
        "invalid_last_edge": tf.constant([[1, 0.7, -2]], dtype=tf.float64),
        "invalid_patial_edges": tf.constant([[-1, 0.7, -2]], dtype=tf.float64),
    }

    return {
        "eom_struts": eom_struts,
    }

@pytest.fixture(scope="module")
def expected_outputs() -> dict:
    eom_struts = {
        "5-cell": {
            "invalid_config": tf.constant(float("nan"), dtype=tf.float64),
            "working_edge_small_diff": tf.constant(644.9876815646884, dtype=tf.float64),
            "working_large_struts": tf.constant(1511.3569100630182, dtype=tf.float64),
            "working_large_spatial_edges": tf.constant(10239182.378913011, dtype=tf.float64),
            "invalid_first_edge": tf.constant([[]], dtype=tf.float64),
            "invalid_strut": tf.constant([[]], dtype=tf.float64),
            "invalid_last_edge": tf.constant([[]], dtype=tf.float64),
            "invalid_patial_edges": tf.constant([[]], dtype=tf.float64),
        },
        "16-cell": {
            "invalid_config": tf.constant(float("nan"), dtype=tf.float64),
            "working_edge_small_diff": tf.constant(1024.1152603514251, dtype=tf.float64),
            "working_large_struts": tf.constant(2392.47887236898, dtype=tf.float64),
            "working_large_spatial_edges": tf.constant(268184.5232615402, dtype=tf.float64),
            "invalid_first_edge": tf.constant([[]], dtype=tf.float64),
            "invalid_strut": tf.constant([[]], dtype=tf.float64),
            "invalid_last_edge": tf.constant([[]], dtype=tf.float64),
            "invalid_patial_edges": tf.constant([[]], dtype=tf.float64),
        },
        "600-cell": {
            "invalid_config": tf.constant(float("nan"), dtype=tf.float64),
            "working_edge_small_diff": tf.constant(14052480.333229035, dtype=tf.float64),
            "working_large_struts": tf.constant(18412.810957911395, dtype=tf.float64),
            "working_large_spatial_edges": tf.constant(68362755375.17299, dtype=tf.float64),
            "invalid_first_edge": tf.constant([[]], dtype=tf.float64),
            "invalid_strut": tf.constant([[]], dtype=tf.float64),
            "invalid_last_edge": tf.constant([[]], dtype=tf.float64),
            "invalid_patial_edges": tf.constant([[]], dtype=tf.float64),
        },
    }

    return {
        "eom_struts": eom_struts,
    }

@pytest.fixture(scope="module")
def model_params() -> dict[str, Model]:
    triangulations = ["5-cell", "16-cell", "600-cell"]
    cosmological_constant = 1e-5
    return {
        triangulation: set_model_params(triangulation, cosmological_constant)
        for triangulation in triangulations
    }

@pytest.mark.parametrize(
    ("case", "triangulation"),
    [
        ("invalid_config", "5-cell"),
        ("working_edge_small_diff", "5-cell"),
        ("working_large_struts", "5-cell"),
        ("working_large_spatial_edges", "5-cell"),
        ("invalid_first_edge", "5-cell"),
        ("invalid_strut", "5-cell"),
        ("invalid_last_edge", "5-cell"),
        ("invalid_patial_edges", "5-cell"),
        ("invalid_config", "16-cell"),
        ("working_edge_small_diff", "16-cell"),
        ("working_large_struts", "16-cell"),
        ("working_large_spatial_edges", "16-cell"),
        ("invalid_first_edge", "16-cell"),
        ("invalid_strut", "16-cell"),
        ("invalid_last_edge", "16-cell"),
        ("invalid_patial_edges", "16-cell"),
        ("invalid_config", "600-cell"),
        ("working_edge_small_diff", "600-cell"),
        ("working_large_struts", "600-cell"),
        ("working_large_spatial_edges", "600-cell"),
        ("invalid_first_edge", "600-cell"),
        ("invalid_strut", "600-cell"),
        ("invalid_last_edge", "600-cell"),
        ("invalid_patial_edges", "600-cell"),
    ]
)
def test_eom_struts(case: str, triangulation: str, inputs: dict, expected_outputs: dict, model_params: dict[str, Model]) -> None:
    function_name = "eom_struts"
    predictions = inputs[function_name]
    results = expected_outputs[function_name][triangulation]

    if case == "invalid_config":
        result = eom_struts(predictions[case], model_params[triangulation])
        assert isinstance(result, tf.Tensor)
        assert tf.math.is_nan(result.numpy())
    elif case == "invalid_first_edge":
        with pytest.raises(tf.errors.InvalidArgumentError,
        match=r"Parameter l1 must not be negative."):
            eom_struts(predictions[case], model_params[triangulation])
    elif case == "invalid_strut":
        with pytest.raises(tf.errors.InvalidArgumentError,
        match=r"Parameter m1 must not be negative."):
            eom_struts(predictions[case], model_params[triangulation])
    elif case == "invalid_last_edge":
        with pytest.raises(tf.errors.InvalidArgumentError,
        match=r"Parameter l2 must not be negative."):
            eom_struts(predictions[case], model_params[triangulation])
    elif case == "invalid_patial_edges":
        with pytest.raises(tf.errors.InvalidArgumentError,
        match=r"Parameter l1 must not be negative."):
            eom_struts(predictions[case], model_params[triangulation])
    else:
        result = eom_struts(predictions[case], model_params[triangulation])
        assert isinstance(result, tf.Tensor)
        tf.debugging.assert_near(
            result,
            results[case],
            rtol=1e-9,
            atol=1e-13
        )


def test_arg1() -> None:
    assert False


def test_arg2() -> None:
    assert False


def test_eoml1() -> None:
    assert False


def test_eoml_acoses() -> None:
    assert False


def test_eoml2() -> None:
    assert False


def test_eom_spatial_edges() -> None:
    assert False
