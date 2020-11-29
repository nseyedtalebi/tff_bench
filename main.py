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

FLAGS = flags.FLAGS


def main(argv):

    # TODO: choose training_process based on args, same for other args to training_loop.run
    # for example: training_process = iterative_process_builder(tff_model_fn)

    logging.info("Training model:")
    logging.info(model_builder().summary())

    #
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
