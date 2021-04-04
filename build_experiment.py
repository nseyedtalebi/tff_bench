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


def build_federated_process(
    model_fn, process_type="fedavg"
) -> tff.templates.IterativeProcess:
    if FLAGS.uniform_weighting:

        def client_weight_fn(local_outputs):
            del local_outputs
            return 1.0

    else:
        client_weight_fn = None  #  Defaults to the number of examples per client.

    if FLAGS.noise_multiplier:
        if not FLAGS.uniform_weighting:
            raise ValueError(
                "Differential privacy is only implemented for uniform weighting."
            )
        aggregation_process = build_dp_aggregate_process(model_fn)
        print("Build dp_aggregate_process")
    else:
        aggregation_process = None

    if process_type == "fedavg":
        process_builder_fn = tff.learning.build_federated_averaging_process
    elif process_type == "fedsgd":
        process_builder_fn = tff.learning.build_federated_sgd_process
    else:
        raise ArgumentError(
            f"Unknown process_type {process_type}. process_type must be 'fedavg' or 'fedsgd'"
        )

    server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags("server")
    client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags("client")

    # add stuff to set these to include differnet things
    # model_update_aggregation_factory=None,
    # broadcast_process=None,

    return process_builder_fn(
        model_fn,
        client_optimizer_fn,
        server_optimizer_fn,
        client_weighting=client_weight_fn,
        broadcast_process=None,
        aggregation_process=aggregation_process,
        model_update_aggregation_factory=None,
        use_experimental_simulation_loop=False,
    )

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

    return tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_weighting=client_weighting,
        client_optimizer_fn=client_optimizer_fn,
        model_update_aggregation_factory=aggregation_factory)

  task_spec = training_specs.TaskSpec(
      iterative_process_builder=iterative_process_builder,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      client_batch_size=FLAGS.client_batch_size,
      clients_per_round=FLAGS.clients_per_round,
      client_datasets_random_seed=FLAGS.client_datasets_random_seed)
