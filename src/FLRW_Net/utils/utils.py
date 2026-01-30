"""Utility functions for FLRW-Net."""

from typing import NamedTuple

import tensorflow as tf


class Triangulation(NamedTuple):
    """Tensorflow graph-execution compatible triangulation params."""
    n1: tf.Tensor
    n2: tf.Tensor
    n3: tf.Tensor
    nte: tf.Tensor

class Model(NamedTuple):
    """Tensorflow graph-execution compatible model params."""
    n1: tf.Tensor
    n2: tf.Tensor
    n3: tf.Tensor
    nte: tf.Tensor
    lamb: tf.Tensor

def get_triangulation_params(triangulation: str) -> Triangulation:
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
        "5-cell": Triangulation(**params_5_cell),
        "16-cell": Triangulation(**params_16_cell),
        "600-cell": Triangulation(**params_600_cell),
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


def set_slice_specs(n: int) -> list[tuple[int, int]]:
    """Compute the slice specs from the number of time steps."""
    length = 2*n-1
    specs = []

    # First element
    specs.append((0, 1))

    # Sliding windows of size 3
    specs.extend((center-1, center+2) for center in range(1, length+1, 2))

    if length >= 2:  # noqa: PLR2004
        # Sliding windows of size 5
        specs.extend((center-2, center+3) for center in range(2, length, 2))

    # Last element
    specs.append((length+1, length+2))

    return specs
