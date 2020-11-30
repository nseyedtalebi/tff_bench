"""Federated EMNIST character recognition library using TFF."""

import functools
from typing import Callable, Optional

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from utils import training_loop
from utils import training_utils
from utils.datasets import emnist_dataset
from utils.models import emnist_models

EMNIST_MODELS = ['cnn', '2nn']

  emnist_train, _ = emnist_dataset.get_emnist_datasets(
      client_batch_size,
      client_epochs_per_round,
      max_batches_per_client=max_batches_per_client,
      only_digits=False)

  _, emnist_test = emnist_dataset.get_centralized_datasets(
      train_batch_size=client_batch_size,
      max_test_batches=max_eval_batches,
      only_digits=False)

  input_spec = emnist_train.create_tf_dataset_for_client(
      emnist_train.client_ids[0]).element_spec

  if model == 'cnn':
    model_builder = functools.partial(
        emnist_models.create_conv_dropout_model, only_digits=False)
  elif model == '2nn':
    model_builder = functools.partial(
        emnist_models.create_two_hidden_layer_model, only_digits=False)
  else:
    raise ValueError(
        'Cannot handle model flag [{!s}], must be one of {!s}.'.format(
            model, EMNIST_MODELS))

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  training_process = iterative_process_builder(tff_model_fn)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      dataset=emnist_train,
      clients_per_round=clients_per_round,
      random_seed=client_datasets_random_seed)

  evaluate_fn = training_utils.build_centralized_evaluate_fn(
      eval_dataset=emnist_test,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  logging.info('Training model:')
  logging.info(model_builder().summary())
