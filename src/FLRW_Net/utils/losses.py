from typing import Callable

import tensorflow as tf

from FLRW_Net.utils.eoms import eom_spatial_edges, eom_struts
from FLRW_Net.utils.utils import Model


@tf.function
def sliding_slice_losses(  # noqa: PLR0913
    prediction: tf.Tensor,
    model_params: Model,
    slice_length: int,
    step: int,
    start: int,
    stop: int,
    loss_fn: Callable[[tf.Tensor], tf.Tensor],
) -> tf.Tensor:
    """Compute the losses for a given loss function."""
    slices = tf.signal.frame(
        prediction,
        frame_length=slice_length,
        frame_step=step,
        axis=1
    )[:, start:stop]

    batch_size = tf.shape(slices)[0]
    number_of_slices = tf.shape(slices)[1]

    slices = tf.reshape(slices, (batch_size * number_of_slices, slice_length))
    losses = loss_fn(slices, model_params)

    return tf.reshape(losses, (batch_size, number_of_slices))

@tf.function
def spatial_edge_losses(prediction: tf.Tensor, model_params: Model) -> tf.Tensor:
    """Compute the tensor of losses from eom_spatial_edges."""
    num_edges = (tf.shape(prediction)[1] - 3) // 2

    return sliding_slice_losses(
        prediction=prediction,
        model_params=model_params,
        slice_length=5,
        step=2,
        start=1,
        stop=1 + num_edges,
        loss_fn=eom_spatial_edges, # type: ignore
    )

@tf.function
def strut_losses(prediction: tf.Tensor, model_params: Model) -> tf.Tensor:
    """Compute the tensor of losses from eom_struts."""
    num_struts = (tf.shape(prediction)[1] - 1) // 2

    return sliding_slice_losses(
        prediction=prediction,
        model_params=model_params,
        slice_length=3,
        step=2,
        start=0,
        stop=num_struts,
        loss_fn=eom_struts, # type: ignore
    )
