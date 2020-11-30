"""
Runs federated training on various tasks using a generalized form of FedAvg.

Specifically, we create (according to flags) an iterative processes that allows
for client and server learning rate schedules, as well as various client and
server optimization methods. For more details on the learning rate scheduling
and optimization methods, see `shared/optimizer_utils.py`. For details on the
iterative process, see `shared/fed_avg_schedule.py`.
"""

import collections
from typing import Any, Callable, Optional

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from grfu.utils import utils_impl
import grfu.utils.training_loop as training_loop
from grfu.optimization import fed_avg_schedule
from grfu.optimization import optimizer_utils

import define_flags

FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError(
            "Expected no command-line arguments, " "got: {}".format(argv)
        )
    from build_experiment import get_training_loop_kwargs

    model_kwargs = get_training_loop_kwargs()
    training_loop_flags = lookup_flag_values(training_loop_flags)
    training_loop_kwargs = {**model_kwargs, **training_loop_flags}
    training_loop.run(**training_loop_kwargs)


if __name__ == "__main__":
    app.run(main)
