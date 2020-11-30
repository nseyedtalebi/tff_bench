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

with utils_impl.record_hparam_flags() as dp_flags:
    # Differential privacy flags
    flags.DEFINE_float("clip", 0.05, "Initial clip.")
    flags.DEFINE_float(
        "noise_multiplier",
        None,
        "Noise multiplier. Set to a float > 1 to enable DP",
    )
    flags.DEFINE_float("adaptive_clip_learning_rate", 0, "Adaptive clip learning rate.")
    flags.DEFINE_float("target_unclipped_quantile", 0.5, "Target unclipped quantile.")
    flags.DEFINE_float(
        "clipped_count_budget_allocation",
        0.1,
        "Fraction of privacy budget to allocate for clipped counts.",
    )


with utils_impl.record_hparam_flags() as optimizer_flags:
    # Defining optimizer flags
    optimizer_utils.define_optimizer_flags("client")
    optimizer_utils.define_optimizer_flags("server")
    optimizer_utils.define_lr_schedule_flags("client")
    optimizer_utils.define_lr_schedule_flags("server")


with utils_impl.record_hparam_flags() as shared_flags:
    # Federated training hyperparameters
    flags.DEFINE_integer(
        "client_epochs_per_round",
        1,
        "Number of epochs in the client to take per round.",
    )
    flags.DEFINE_integer("client_batch_size", 20, "Batch size on the clients.")
    flags.DEFINE_integer(
        "clients_per_round", 10, "How many clients to sample per round."
    )
    flags.DEFINE_integer(
        "client_datasets_random_seed", 1, "Random seed for client sampling."
    )

with utils_impl.record_hparam_flags() as training_loop_flags:
    # Training loop configuration
    flags.DEFINE_string(
        "experiment_name",
        None,
        "The name of this experiment. Will be append to "
        "--root_output_dir to separate experiment results.",
    )
    flags.mark_flag_as_required("experiment_name")
    flags.DEFINE_string(
        "root_output_dir",
        "/tmp/fed_opt/",
        "Root directory for writing experiment output.",
    )
    flags.DEFINE_integer("total_rounds", 200, "Number of total training rounds.")
    flags.DEFINE_integer(
        "rounds_per_eval",
        1,
        "How often to evaluate the global model on the validation dataset.",
    )
    flags.DEFINE_integer(
        "rounds_per_checkpoint", 50, "How often to checkpoint the global model."
    )
    flags.DEFINE_integer(
        "rounds_per_profile",
        0,
        "(Experimental) How often to run the experimental TF profiler, if >0.",
    )

from build_experiment import get_training_loop_kwargs


FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError(
            "Expected no command-line arguments, " "got: {}".format(argv)
        )
    model_kwargs = get_training_loop_kwargs()
    training_loop_flags = lookup_flag_values(training_loop_flags)
    training_loop_kwargs = {**model_kwargs, **training_loop_flags}
    training_loop.run(**training_loop_kwargs)


if __name__ == "__main__":
    app.run(main)
