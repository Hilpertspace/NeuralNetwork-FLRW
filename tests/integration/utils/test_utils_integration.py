import pytest

from FLRW_Net.utils.utils import Model, set_model_params


@pytest.fixture(scope="module")
def expected_params() -> dict[str, dict[str, int] | float]:
    cosmological_constant = 1e-5

    params_5_cell = {
        "n1": 10,
        "n2": 10,
        "n3": 5,
        "nte": 3,
        "lamb": cosmological_constant,
    }
    params_16_cell = {
        "n1": 24,
        "n2": 32,
        "n3": 16,
        "nte": 4,
        "lamb": cosmological_constant,
    }
    params_600_cell = {
        "n1": 720,
        "n2": 1200,
        "n3": 600,
        "nte": 5,
        "lamb": cosmological_constant
    }
    return {
        "5-cell": params_5_cell,
        "16-cell": params_16_cell,
        "600-cell": params_600_cell,
        "cosmological_constant": cosmological_constant,
    }

@pytest.mark.parametrize(
    "triangulation",
    [
        "5-cell",
        "16-cell",
        "600-cell",
        "invalid",
    ]
)
def test_set_model_params(triangulation: str, expected_params: dict) -> None:
    cosmological_constant = expected_params["cosmological_constant"]
    if triangulation != "invalid":
        model_params = set_model_params(triangulation, cosmological_constant)
        assert isinstance(model_params, Model)
        assert model_params._asdict() == expected_params[triangulation]
    else:
        with pytest.raises(ValueError, match=f"The given triangulation {triangulation} is invalid. Please choose from '5-cell', '16-cell', '600-cell'."):
            set_model_params(triangulation, cosmological_constant)
