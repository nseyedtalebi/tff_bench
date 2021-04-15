import typing
import collections
from typing import Tuple, Callable

from absl import flags
import tensorflow_federated as tff
import tensorflow as tf
import tensorflow_privacy as tp
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

import utils.utils_impl as utils_impl
from utils.datasets import emnist_dataset
from utils.models import emnist_models
from utils.utils_impl import lookup_flag_values
from optimization.shared import optimizer_utils
from optimization.shared.keras_metrics import NumExamplesCounter

FLAGS = flags.FLAGS

def iterative_process_builder(
      model_fn: Callable[[], tff.learning.Model],
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    logging.info('Trainable weights:')
    for weight in model_fn().weights.trainable:
      logging.info('name: %s  shape: %s', weight.name, weight.shape)

    if FLAGS.uniform_weighting:
      client_weighting = tff.learning.ClientWeighting.UNIFORM
    elif FLAGS.task == 'shakespeare' or FLAGS.task == 'stackoverflow_nwp':

      def client_weighting(local_outputs):
        return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)
    else:
      client_weighting = None

    if FLAGS.noise_multiplier is None:
      if FLAGS.uniform_weighting:
        aggregation_factory = tff.aggregators.UnweightedMeanFactory()
      else:
        aggregation_factory = tff.aggregators.MeanFactory()
      if FLAGS.clip is not None:
        if FLAGS.clip <= 0:
          raise ValueError('clip must be positive if clipping is enabled.')
        if FLAGS.adaptive_clip_learning_rate is None:
          clip = FLAGS.clip
        else:
          if FLAGS.adaptive_clip_learning_rate <= 0:
            raise ValueError('adaptive_clip_learning_rate must be positive if '
                             'adaptive clipping is enabled.')
          clip = tff.aggregators.PrivateQuantileEstimationProcess.no_noise(
              initial_estimate=FLAGS.clip,
              target_quantile=FLAGS.target_unclipped_quantile,
              learning_rate=FLAGS.adaptive_clip_learning_rate)
        aggregation_factory = tff.aggregators.clipping_factory(
            clip, aggregation_factory)
    else:
      if not FLAGS.uniform_weighting:
        raise ValueError(
            'Differential privacy is only implemented for uniform weighting.')
      if FLAGS.noise_multiplier <= 0:
        raise ValueError('noise_multiplier must be positive if DP is enabled.')
      if FLAGS.clip is None or FLAGS.clip <= 0:
        raise ValueError('clip must be positive if DP is enabled.')
      if FLAGS.adaptive_clip_learning_rate is None:
        aggregation_factory = tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
            noise_multiplier=FLAGS.noise_multiplier,
            clients_per_round=FLAGS.clients_per_round,
            clip=FLAGS.clip)
      else:
        if FLAGS.adaptive_clip_learning_rate <= 0:
          raise ValueError('adaptive_clip_learning_rate must be positive if '
                           'adaptive clipping is enabled.')
        aggregation_factory = tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive(
            noise_multiplier=FLAGS.noise_multiplier,
            clients_per_round=FLAGS.clients_per_round,
            initial_l2_norm_clip=FLAGS.clip,
            target_unclipped_quantile=FLAGS.target_unclipped_quantile,
            learning_rate=FLAGS.adaptive_clip_learning_rate)

    #add stuff for compression
    if FLAGS.use_compression:
    # We create a `MeasuredProcess` for broadcast process and a
    # `MeasuredProcess` for aggregate process by providing the
    # `_broadcast_encoder_fn` and `_mean_encoder_fn` to corresponding utilities.
    # The fns are called once for each of the model weights created by
    # tff_model_fn, and return instances of appropriate encoders.
    encoded_broadcast_process = (
        tff.learning.framework.build_encoded_broadcast_process_from_model(
            tff_model_fn, _broadcast_encoder_fn))
    encoded_mean_process = (
        tff.learning.framework.build_encoded_mean_process_from_model(
            tff_model_fn, _mean_encoder_fn))
    else:
      encoded_broadcast_process = None
      encoded_mean_process = None

    return tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_weighting=client_weighting,
        client_optimizer_fn=client_optimizer_fn,
        model_update_aggregation_factory=aggregation_factory,
        aggregation_process=encoded_mean_process,
        broadcast_process=encoded_broadcast_proces)

def _broadcast_encoder_fn(value):
  """Function for building encoded broadcast.

  This method decides, based on the tensor size, whether to use lossy
  compression or keep it as is (use identity encoder). The motivation for this
  pattern is due to the fact that compression of small model weights can provide
  only negligible benefit, while at the same time, lossy compression of small
  weights usually results in larger impact on model's accuracy.

  Args:
    value: A tensor or variable to be encoded in server to client communication.

  Returns:
    A `te.core.SimpleEncoder`.
  """
  # TODO(b/131681951): We cannot use .from_tensor(...) because it does not
  # currently support Variables.
  spec = tf.TensorSpec(value.shape, value.dtype)
  if value.shape.num_elements() > 10000:
    return te.encoders.as_simple_encoder(
        te.encoders.uniform_quantization(FLAGS.broadcast_quantization_bits),
        spec)
  else:
    return te.encoders.as_simple_encoder(te.encoders.identity(), spec)


def _mean_encoder_fn(value):
  """Function for building encoded mean.

  This method decides, based on the tensor size, whether to use lossy
  compression or keep it as is (use identity encoder). The motivation for this
  pattern is due to the fact that compression of small model weights can provide
  only negligible benefit, while at the same time, lossy compression of small
  weights usually results in larger impact on model's accuracy.

  Args:
    value: A tensor or variable to be encoded in client to server communication.

  Returns:
    A `te.core.GatherEncoder`.
  """
  # TODO(b/131681951): We cannot use .from_tensor(...) because it does not
  # currently support Variables.
  spec = tf.TensorSpec(value.shape, value.dtype)
  if value.shape.num_elements() > 10000:
    if FLAGS.use_sparsity_in_aggregation:
      return te.encoders.as_gather_encoder(
          sparsity.sparse_quantizing_encoder(
              FLAGS.aggregation_quantization_bits), spec)
    else:
      return te.encoders.as_gather_encoder(
          te.encoders.uniform_quantization(FLAGS.aggregation_quantization_bits),
          spec)
  else:
    return te.encoders.as_gather_encoder(te.encoders.identity(), spec)
