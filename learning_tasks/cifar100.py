"""Federated CIFAR-100 classification library using TFF."""

import functools
from typing import Callable, Optional

from absl import logging
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from grfu.utils import training_utils
from grfu.utils.datasets import cifar100_dataset
from grfu.utils.models import resnet_models
FLAGS = flags.FLAGS

class Cifar100:#eventually, this will derive from an abstract class mb

    CIFAR_SHAPE = (32, 32, 3)
    NUM_CLASSES = 100
    crop_shape = (FLAGS.cifar100_crop_size, FLAGS.cifar100_crop_size, 3)

    def __init__(self, addl_metrics=[]):
        self._train_data = None
        self._test_data = None
        self._input_spec = None
        self._model_builder = None
        self._loss_builder = None
        self._metrics_builder = None
        self._client_datasets = None
        self._evaluate_fn = None
        self._tff_model_fn = None
        self.addl_metrics = addl_metrics

    @property
    def train_data() -> tff.simulation.ClientData:
        if not self._cifar_train:
            self._cifar_train, _ = cifar100_dataset.get_federated_cifar100(
                client_epochs_per_round=FLAGS.client_epochs_per_round,
                train_batch_size=FLAGS.client_batch_size,
                crop_shape=crop_shape,
                max_batches_per_client=-1)
        return self._cifar_train

    @property
    def test_data():
        if not self._cifar_test:
            _, cifar_test = cifar100_dataset.get_centralized_datasets(
                train_batch_size=FLAGS.client_batch_size,
                max_test_batches=None,
                crop_shape=crop_shape)
        return self._cifar_test

    @property
    def input_spec():
        if not self._input_spec:
            self._input_spec = self.train_data.create_tf_dataset_for_client(
                cifar_train.client_ids[0]).element_spec
        return self._input_spec

    @property
    def model_builder():
        if not self._model_builder:
            self._model_builder = functools.partial(
                resnet_models.create_resnet18,
                input_shape=crop_shape,
                num_classes=NUM_CLASSES)
        return self._model_builder

    @property
    def loss_builder():
        if not self._loss_builder:
            self._loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
        return self._loss_builder

    @property
    def metrics_builder():
        if not self._metrics_builder:
            self._metrics_builder = \
                addl_metrics + [tf.keras.metrics.SparseCategoricalAccuracy()]
        return self._metrics_builder

    @property
    def client_datasets_fn():
        if not self._client_datasets_fn:
            self._client_datasets_fn = training_utils.build_client_datasets_fn(
                dataset=self.train_data,
                clients_per_round=FLAGS.clients_per_round,
                random_seed=FLAGS.client_datasets_random_seed
        )
        return self._client_datasets_fn

    @property
    def evaluate_fn():
        if not self._evaluate_fn:
            self._evaluate_fn = training_utils.build_centralized_evaluate_fn(
                eval_dataset=self.test_data,
                model_builder=self.model_builder,
                loss_builder=self.loss_builder,
                metrics_builder=self.metrics_builder
            )

    @property
    def tff_model_fn() -> tff.learning.Model:
        if not self._tff_model_fn:
            self._tff_model_fn = tff.learning.from_keras_model(
                keras_model=self.model_builder,
                input_spec=self.input_spec,
                loss=self.loss_builder,
                metrics=self.metrics_builder
            )
