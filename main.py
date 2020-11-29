import functools
from typing import Callable, Optional

from absl import flags
from absl import logging
from absl import app
import tensorflow as tf
import tensorflow_federated as tff

from grfu.utils import training_loop
from grfu.utils import training_utils
from grfu.utils.datasets import emnist_dataset
from grfu.utils.models import emnist_models

from emnist_experiment import emnistExperiment

emnistExperiment.define_experiment_flags()

FLAGS = flags.FLAGS


def main(argv):
    print(FLAGS.client_optimizer)
    emnist = emnistExperiment()
    client_datasets_fn = None
    validation_fn = None
    test_fn = None
    total_rounds = None
    experiment_name = "emnist_1"
    root_output_dir = "/tmp/emnist_1/"

    loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
    metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

    def tff_model_fn() -> tff.learning.Model:
        return tff.learning.from_keras_model(
            keras_model=model_builder(),
            input_spec=input_spec,
            loss=loss_builder(),
            metrics=metrics_builder(),
        )

    training_process = emnist.get_iterative_process()

    client_datasets_fn = training_utils.build_client_datasets_fn(
        dataset=emnist_train,
        clients_per_round=clients_per_round,
        random_seed=client_datasets_random_seed,
    )

    evaluate_fn = training_utils.build_centralized_evaluate_fn(
        eval_dataset=emnist_test,
        model_builder=model_builder,
        loss_builder=loss_builder,
        metrics_builder=metrics_builder,
    )

    logging.info("Training model:")
    logging.info(model_builder().summary())
    training_loop.run(
        iterative_process=training_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=evaluate_fn,
        test_fn=evaluate_fn,
        total_rounds=total_rounds,
        experiment_name=experiment_name,
        root_output_dir=root_output_dir,
        **kwargs
    )


if __name__ == "__main__":
    app.run(main)
