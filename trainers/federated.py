# Copyright 2020, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs federated training on various tasks using a generalized form of FedAvg.

Specifically, we create (according to flags) an iterative processes that allows
for client and server learning rate schedules, as well as various client and
server optimization methods. For more details on the learning rate scheduling
and optimization methods, see `shared/optimizer_utils.py`. For details on the
iterative process, see `shared/fed_avg_schedule.py`.
"""

import collections
import os.path
from typing import Callable

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from optimization.cifar100 import federated_cifar100
from optimization.emnist import federated_emnist
from optimization.emnist_ae import federated_emnist_ae
from optimization.shakespeare import federated_shakespeare
from optimization.shared import fed_avg_schedule
from optimization.shared import optimizer_utils
from optimization.shared import training_specs
from optimization.stackoverflow import federated_stackoverflow
from optimization.stackoverflow_lr import federated_stackoverflow_lr
from utils import training_loop
from utils import utils_impl

_SUPPORTED_TASKS = [
    'cifar100', 'emnist_cr', 'emnist_ae', 'shakespeare', 'stackoverflow_nwp',
    'stackoverflow_lr'
]

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')
  optimizer_utils.define_lr_schedule_flags('client')
  optimizer_utils.define_lr_schedule_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')

  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

with utils_impl.record_hparam_flags() as task_flags:
  # Task specification
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

with utils_impl.record_hparam_flags() as cifar100_flags:
  # CIFAR-100 flags
  flags.DEFINE_integer('cifar100_crop_size', 24, 'The height and width of '
                       'images after preprocessing.')
  flags.DEFINE_bool(
      'cifar100_distort_train_images', True, 'If set to True, '
      'train images will be randomly cropped. Otherwise, all '
      'images will simply be resized.')

with utils_impl.record_hparam_flags() as emnist_cr_flags:
  # EMNIST CR flags
  flags.DEFINE_enum(
      'emnist_cr_model', 'cnn', ['cnn', '2nn'], 'Which model to '
      'use. This can be a convolutional model (cnn) or a two '
      'hidden-layer densely connected network (2nn).')

with utils_impl.record_hparam_flags() as shakespeare_flags:
  # Shakespeare flags
  flags.DEFINE_integer(
      'shakespeare_sequence_length', 80,
      'Length of character sequences to use for the RNN model.')

with utils_impl.record_hparam_flags() as so_nwp_flags:
  # Stack Overflow NWP flags
  flags.DEFINE_integer('so_nwp_vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('so_nwp_num_oov_buckets', 1,
                       'Number of out of vocabulary buckets.')
  flags.DEFINE_integer('so_nwp_sequence_length', 20,
                       'Max sequence length to use.')
  flags.DEFINE_integer('so_nwp_max_elements_per_user', 1000, 'Max number of '
                       'training sentences to use per user.')
  flags.DEFINE_integer(
      'so_nwp_num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')

with utils_impl.record_hparam_flags() as so_lr_flags:
  # Stack Overflow LR flags
  flags.DEFINE_integer('so_lr_vocab_tokens_size', 10000,
                       'Vocab tokens size used.')
  flags.DEFINE_integer('so_lr_vocab_tags_size', 500, 'Vocab tags size used.')
  flags.DEFINE_integer(
      'so_lr_num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')
  flags.DEFINE_integer('so_lr_max_elements_per_user', 1000,
                       'Max number of training '
                       'sentences to use per user.')

with utils_impl.record_hparam_flags() as dp_flags:
  # Differential privacy flags
  flags.DEFINE_float(
      'clip', None, 'Clip value for fixed clipping or initial clip for '
      'adaptive clipping. If None, no clipping is used.')
  flags.DEFINE_float('noise_multiplier', None,
                     'Noise multiplier. If None, non-DP aggregator is used.')
  flags.DEFINE_float(
      'adaptive_clip_learning_rate', None, 'Adaptive clip learning rate. If '
      'None, clip adaptation is not used.')
  flags.DEFINE_float('target_unclipped_quantile', 0.5,
                     'Target unclipped quantile.')
  flags.DEFINE_boolean('uniform_weighting', False,
                       'Whether to weigh clients uniformly.')

# adding adaptive lr and making it compatible with the other options will take some time
# I need to do quicker things first if I can
#with utils_impl.record_hparam_flags() as callback_flags:
#  flags.DEFINE_float(
#      'client_decay_factor', 0.1, 'Amount to decay the client learning rate '
#      'upon reaching a plateau.')
#  flags.DEFINE_float(
#      'server_decay_factor', 0.9, 'Amount to decay the server learning rate '
#      'upon reaching a plateau.')
#  flags.DEFINE_float(
#      'min_delta', 1e-4, 'Minimum delta for improvement in the learning rate '
#      'callbacks.')
#  flags.DEFINE_integer(
#      'window_size', 100, 'Number of rounds to take a moving average over when '
#      'estimating the training loss in learning rate callbacks.')
#  flags.DEFINE_integer(
#      'patience', 100, 'Number of rounds of non-improvement before decaying the'
#      'learning rate.')
#  flags.DEFINE_float('min_lr', 0.0, 'The minimum learning rate.')

  # Compression hyperparameters.
with utils.impl.record_hparam_flags() as compression_flags:
  flags.DEFINE_boolean('use_compression', True,
                       'Whether to use compression code path.')
  flags.DEFINE_integer(
      'broadcast_quantization_bits', 8,
      'Number of quantization bits for server to client '
      'compression.')
  flags.DEFINE_integer(
      'aggregation_quantization_bits', 8,
      'Number of quantization bits for client to server '
      'compression.')
  flags.DEFINE_boolean('use_sparsity_in_aggregation', True,
                       'Whether to add sparsity to the aggregation. This will '
                       'only be used for client to server compression.')
FLAGS = flags.FLAGS

TASK_FLAGS = collections.OrderedDict(
    cifar100=cifar100_flags,
    emnist_cr=emnist_cr_flags,
    shakespeare=shakespeare_flags,
    stackoverflow_nwp=so_nwp_flags,
    stackoverflow_lr=so_lr_flags)


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task-specific flags.
  task_name = FLAGS.task
  if task_name in TASK_FLAGS:
    task_hparam_dict = utils_impl.lookup_flag_values(TASK_FLAGS[task_name])
    hparam_dict.update(task_hparam_dict)

  results_dir = os.path.join(FLAGS.root_output_dir, 'results',
                             FLAGS.experiment_name)
  utils_impl.create_directory_if_not_exists(results_dir)
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  utils_impl.atomic_write_series_to_csv(hparam_dict, hparam_file)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  client_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('client')
  server_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('server')

  task_spec = training_specs.TaskSpec(
      iterative_process_builder=iterative_process_builder,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      client_batch_size=FLAGS.client_batch_size,
      clients_per_round=FLAGS.clients_per_round,
      client_datasets_random_seed=FLAGS.client_datasets_random_seed)
  


  if FLAGS.task == 'cifar100':
    runner_spec = federated_cifar100.configure_training(
        task_spec,
        crop_size=FLAGS.cifar100_crop_size,
        distort_train_images=FLAGS.cifar100_distort_train_images)
  elif FLAGS.task == 'emnist_cr':
    runner_spec = federated_emnist.configure_training(
        task_spec, model=FLAGS.emnist_cr_model)
  elif FLAGS.task == 'emnist_ae':
    runner_spec = federated_emnist_ae.configure_training(task_spec)
  elif FLAGS.task == 'shakespeare':
    runner_spec = federated_shakespeare.configure_training(
        task_spec, sequence_length=FLAGS.shakespeare_sequence_length)
  elif FLAGS.task == 'stackoverflow_nwp':
    runner_spec = federated_stackoverflow.configure_training(
        task_spec,
        vocab_size=FLAGS.so_nwp_vocab_size,
        num_oov_buckets=FLAGS.so_nwp_num_oov_buckets,
        sequence_length=FLAGS.so_nwp_sequence_length,
        max_elements_per_user=FLAGS.so_nwp_max_elements_per_user,
        num_validation_examples=FLAGS.so_nwp_num_validation_examples)
  elif FLAGS.task == 'stackoverflow_lr':
    runner_spec = federated_stackoverflow_lr.configure_training(
        task_spec,
        vocab_tokens_size=FLAGS.so_lr_vocab_tokens_size,
        vocab_tags_size=FLAGS.so_lr_vocab_tags_size,
        max_elements_per_user=FLAGS.so_lr_max_elements_per_user,
        num_validation_examples=FLAGS.so_lr_num_validation_examples)
  else:
    raise ValueError(
        '--task flag {} is not supported, must be one of {}.'.format(
            FLAGS.task, _SUPPORTED_TASKS))

  _write_hparam_flags()


  training_loop.run(
      iterative_process=runner_spec.iterative_process,
      client_datasets_fn=runner_spec.client_datasets_fn,
      validation_fn=runner_spec.validation_fn,
      test_fn=runner_spec.test_fn,
      total_rounds=FLAGS.total_rounds,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      rounds_per_eval=FLAGS.rounds_per_eval,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)


if __name__ == '__main__':
  app.run(main)