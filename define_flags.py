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

#Task flags
with utils_impl.record_hparam_flags() as cifar100_flags:
    # CIFAR-100 flags
    flags.DEFINE_integer(
        "cifar100_crop_size",
        24,
        "The height and width of " "images after preprocessing.",
    )

with utils_impl.record_hparam_flags() as emnist_cr_flags:
    # EMNIST CR flags
    flags.DEFINE_enum(
        "emnist_cr_model",
        "cnn",
        ["cnn", "2nn"],
        "Which model to "
        "use. This can be a convolutional model (cnn) or a two "
        "hidden-layer densely connected network (2nn).",
    )

with utils_impl.record_hparam_flags() as shakespeare_flags:
    # Shakespeare flags
    flags.DEFINE_integer(
        "shakespeare_sequence_length",
        80,
        "Length of character sequences to use for the RNN model.",
    )

with utils_impl.record_hparam_flags() as so_nwp_flags:
    # Stack Overflow NWP flags
    flags.DEFINE_integer("so_nwp_vocab_size", 10000, "Size of vocab to use.")
    flags.DEFINE_integer(
        "so_nwp_num_oov_buckets", 1, "Number of out of vocabulary buckets."
    )
    flags.DEFINE_integer("so_nwp_sequence_length", 20, "Max sequence length to use.")
    flags.DEFINE_integer(
        "so_nwp_max_elements_per_user",
        1000,
        "Max number of " "training sentences to use per user.",
    )
    flags.DEFINE_integer(
        "so_nwp_num_validation_examples",
        10000,
        "Number of examples " "to use from test set for per-round validation.",
    )
    flags.DEFINE_integer(
        "so_nwp_embedding_size", 96, "Dimension of word embedding to use."
    )
    flags.DEFINE_integer(
        "so_nwp_latent_size", 670, "Dimension of latent size to use in recurrent cell"
    )
    flags.DEFINE_integer(
        "so_nwp_num_layers", 1, "Number of stacked recurrent layers to use."
    )
    flags.DEFINE_boolean(
        "so_nwp_shared_embedding",
        False,
        "Boolean indicating whether to tie input and output embeddings.",
    )

with utils_impl.record_hparam_flags() as so_lr_flags:
    # Stack Overflow LR flags
    flags.DEFINE_integer("so_lr_vocab_tokens_size", 10000, "Vocab tokens size used.")
    flags.DEFINE_integer("so_lr_vocab_tags_size", 500, "Vocab tags size used.")
    flags.DEFINE_integer(
        "so_lr_num_validation_examples",
        10000,
        "Number of examples " "to use from test set for per-round validation.",
    )
    flags.DEFINE_integer(
        "so_lr_max_elements_per_user",
        1000,
        "Max number of training " "sentences to use per user.",
    )

SUPPORTED_TASKS = [
    "cifar100",
    "emnist_cr",
    "emnist_ae",
    "shakespeare",
    "stackoverflow_nwp",
    "stackoverflow_lr",
]

FLAGS = flags.FLAGS

TASK_FLAGS = collections.OrderedDict(
    cifar100=lookup_flag_values(cifar100_flags),
    emnist_cr=lookup_flag_values(emnist_cr_flags),
    shakespeare=lookup_flag_values(shakespeare_flags),
    stackoverflow_nwp=lookup_flag_values(so_nwp_flags),
    stackoverflow_lr=lookup_flag_values(so_lr_flags),
)
TASK_FLAG_PREFIXES = collections.OrderedDict(
    cifar100="cifar100",
    emnist_cr="emnist_cr",
    emnist_ae="emnist_ae",
    shakespeare="shakespeare",
    stackoverflow_nwp="so_nwp",
    stackoverflow_lr="so_lr",
)
with utils_impl.record_hparam_flags() as task_flags:
    # Task specification
    flags.DEFINE_enum(
        "task", None, SUPPORTED_TASKS, "Which task to perform federated training on."
    )
def _get_hparam_flags():
    """Returns an ordered dictionary of pertinent hyperparameter flags."""
    hparam_dict = utils_impl.lookup_flag_values(shared_flags)

    # Update with optimizer flags corresponding to the chosen optimizers.
    opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
    opt_flag_dict = optimizer_utils.remove_unused_flags("client", opt_flag_dict)
    opt_flag_dict = optimizer_utils.remove_unused_flags("server", opt_flag_dict)
    hparam_dict.update(opt_flag_dict)

    # Update with task-specific flags.
    task_name = FLAGS.task
    if task_name in TASK_FLAGS:
        task_hparam_dict = utils_impl.lookup_flag_values(TASK_FLAGS[task_name])
        hparam_dict.update(task_hparam_dict)

    return hparam_dict


def _get_task_args():
    """Returns an ordered dictionary of task-specific arguments.

    This method returns a dict of (arg_name, arg_value) pairs, where the
    arg_name has had the task name removed as a prefix (if it exists), as well
    as any leading `-` or `_` characters.

    Returns:
      An ordered dictionary of (arg_name, arg_value) pairs.
    """
    task_name = FLAGS.task
    task_args = collections.OrderedDict()

    if task_name in TASK_FLAGS:
        task_flag_list = TASK_FLAGS[task_name]
        task_flag_dict = utils_impl.lookup_flag_values(task_flag_list)
        task_flag_prefix = TASK_FLAG_PREFIXES[task_name]
        for (key, value) in task_flag_dict.items():
            if key.startswith(task_flag_prefix):
                key = key[len(task_flag_prefix) :].lstrip("_-")
            task_args[key] = value
    return task_args
