import typing
import collections
from typing import Tuple, Callable

from absl import flags
import tensorflow_federated as tff
import tensorflow as tf
import tensorflow_privacy as tp
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

import grfu.utils.utils_impl as utils_impl
from grfu.utils.datasets import emnist_dataset
from grfu.utils.models import emnist_models
from grfu.utils.utils_impl import lookup_flag_values
from grfu.optimization import optimizer_utils
from grfu.optimization.keras_metrics import  NumExamplesCounter

import learning_tasks.cifar100 as cifar100
FLAGS = flags.FLAGS

def build_federated_process(
    model_fn,
    process_type="fedavg",#fedavg or fedsgd
    client_weight_fn=1.0
) -> tff.templates.IterativeProcess:
    if process_type == "fedavg":
        process_builder_fn = tff.learning.build_federated_averaging_process
    elif process_type == "fedsgd":
        process_builder_fn = tff.learning.build_federated_sgd_process
    else:
        raise ArgumentError(f"Unknown process_type {process_type}. process_type must be 'fedavg' or 'fedsgd'")

    server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
    client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')

    if FLAGS.noise_multiplier:
        if not FLAGS.uniform_weighting:
            raise ValueError('Differential privacy is only implemented for uniform weighting.')
        dp_flags =  util_impl.lookup_flag_values("dp_flags")
        aggregation_process = build_dp_aggregate_process(model_fn, **dp_flags)
    else:
        aggregation_process = None

    #add stuff to set these to include differnet things
    #model_update_aggregation_factory=None,
    #broadcast_process=None,

    return process_builder_fn(
        model_fn,
        client_optimizer_fn,
        server_optimizer_fn,
        client_weight_fn,
        broadcast_process=None,
        aggregation_process=aggregation_process,
        model_update_aggregation_factory=None,
        use_experimental_simulation_loop=False
    )

class CalculateDPEpsilon(tf.keras.metrics.Sum):
    def __init__(self, num_examples_all_clients, epochs, name: str = 'dp_epsilon', dtype=tf.float64):  # pylint: disable=useless-super-delegation
        self.num_examples_all_clients = num_examples_all_clients
        self.epochs = epochs
        super().__init__(name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        num_examples = tf.shape(y_pred)[0]
        q = num_examples / self.num_examples_all_clients
        steps = int(math.ceil(self.epochs * self.num_examples_all_clients / num_examples))
        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                    list(range(5, 64)) + [128, 256, 512])
        rdp = compute_rdp(q, noise_multiplier, steps, orders)
        eps, _, opt_order = get_privacy_spent(orders, rdp, target_eps=None, target_delta=None)
        return super().update_state(eps)

'''
can subclass a sum metric to get epsilon
can calculate epsilon after the fact using metrics output
def get_dp_metric():
    q = batch_size / n  # q - the sampling ratio.
    if q > 1:
        raise app.UsageError('n must be larger than the batch size.')
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                list(range(5, 64)) + [128, 256, 512])
    steps = int(math.ceil(epochs * n / batch_size))

    def compute_rdp(q, noise_multiplier, steps, orders):
  """Compute RDP of the Sampled Gaussian Mechanism.
  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
        to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders, can be np.inf.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp(q, noise_multiplier, orders)
  else:
    rdp = np.array([_compute_rdp(q, noise_multiplier, order)
                    for order in orders])

  return rdp * steps


def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  """Compute delta (or eps) for given eps (or delta) from RDP values.
  Args:
    orders: An array (or a scalar) of RDP orders.
    rdp: An array of RDP values. Must be of the same length as the orders list.
    target_eps: If not None, the epsilon for which we compute the corresponding
              delta.
    target_delta: If not None, the delta for which we compute the corresponding
              epsilon. Exactly one of target_eps and target_delta must be None.
  Returns:
    eps, delta, opt_order.
  Raises:
    ValueError: If target_eps and target_delta are messed up.
  """

  return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)
 '''
def build_dp_aggregate_process(model_fn,
    clip = 0.05,
    noise_multiplier = 1.1,#From an example for tensorflow_privacy
    adaptive_clip_learning_rate = 0,
    target_unclipped_quantile = 0.5,
    clipped_count_budget_allocation = 0.1
) -> tff.templates.MeasuredProcess:
    dp_query = tff.utils.build_dp_query(
        clip,
        noise_multiplier,
        adaptive_clip_learning_rate,
        target_unclipped_quantile,
        clipped_count_budget_allocation
    )
    '''
    Args for PrivacyLedger:
    population_size: An integer (may be variable) specifying the size of the
        population, i.e. size of the training data used in each epoch.
      selection_probability: A float (may be variable) specifying the
        probability each record is included in a sample.
    '''
    #examples_per_client = 100#for cifar100
    #FLAGS.client_batch_size * FLAGS.clients_per_round * examples_per_client
    #population_size =
    #selection_probability =
    #ledger = tp.privacy.privacy_ledger.PrivacyLedger(population_size=None, selection_probability=None)
    weights_type = tff.learning.framework.weights_type_from_model(model_fn)
    return tff.utils.build_dp_aggregate_process(weights_type.trainable, dp_query)

def get_training_loop_kwargs():

    if FLAGS.task == "cifar100":

        num_train_examples = 500000 #from tff documentation
        epochs = FLAGS.client_epochs_per_round
        dp_eps_metrics = [CalculateDPEpsilon(num_examples_all_clients=num_train_examples, epochs=epochs)]
        cifar = cifar100.Cifar100(dp_eps_metrics)
        model_fn = cifar.model_fn
        iterative_process = build_federated_process(model_fn)
        client_datasets_fn = cifar.client_datasets_fn
        validation_fn = cifar.evaluate_fn
        test_fn = cifar.evaluate_fn


    elif FLAGS.task == "emnist_cr":
        raise NotImplementedError
    elif FLAGS.task == "emnist_ae":
        raise NotImplementedError
    elif FLAGS.task == "shakespeare":
        raise NotImplementedError
    elif FLAGS.task == "stackoverflow_nwp":
        raise NotImplementedError
    elif FLAGS.task == "stackoverflow_lr":
        raise NotImplementedError
    else:
        raise ValueError(
            "--task flag {} is not supported, must be one of {}.".format(
                FLAGS.task, SUPPORTED_TASKS
            )
        )
    return iterative_process, client_datasets_fn, validation_fn, test_fn
