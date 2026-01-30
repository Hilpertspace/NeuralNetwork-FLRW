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
    def fn(slice_: tf.Tensor) -> tf.Tensor:
        return loss_fn(slice_, model_params)

    slices = tf.signal.frame(
        prediction,
        frame_length=slice_length,
        frame_step=step,
        axis=1
    )[:, start:stop]
    slices = tf.transpose(slices, perm=[1, 0, 2])
    losses = tf.map_fn(fn, slices, fn_output_signature=tf.float64)

    return tf.expand_dims(losses, axis=0)

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
