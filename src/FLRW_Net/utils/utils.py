"""Utility functions for FLRW-Net."""

import tensorflow as tf


def get_triangulation_params(triangulation: str) -> dict[str, int]:
    """Set the parameters that specify the spatial triangulations."""
    params_5_cell = {
        "n1": 10,
        "n2": 10,
        "n3": 5,
        "nte": 3,
    }

    params_16_cell = {
        "n1": 24,
        "n2": 32,
        "n3": 16,
        "nte": 4,
    }

    params_600_cell = {
        "n1": 720,
        "n2": 1200,
        "n3": 600,
        "nte": 5,
    }

    triangulations = {
        "5-cell": params_5_cell,
        "16-cell": params_16_cell,
        "600-cell": params_600_cell,
    }

    if triangulation not in triangulations:
        msg = f"The given triangulation {triangulation} is invalid. Please choose from '5-cell', '16-cell', '600-cell'."
        raise ValueError(msg)

    return triangulations[triangulation]

@tf.function
def crop(arg: tf.Tensor) -> tf.Tensor:
    """Clamp values to the valid domain of tf.acos: [-1, 1]."""
    return tf.clip_by_value(
        arg,
        tf.constant(-1.0, dtype=arg.dtype),
        tf.constant(1.0, dtype=arg.dtype),
    )
