
_SUPPORTED_TASKS = [
    "cifar100",
    "emnist_cr",
    "emnist_ae",
    "shakespeare",
    "stackoverflow_nwp",
    "stackoverflow_lr",
]

with utils_impl.record_hparam_flags() as task_flags:
    # Task specification
    flags.DEFINE_enum(
        "task", None, _SUPPORTED_TASKS, "Which task to perform federated training on."
    )

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

FLAGS = flags.FLAGS

TASK_FLAGS = collections.OrderedDict(
    cifar100=cifar100_flags,
    emnist_cr=emnist_cr_flags,
    shakespeare=shakespeare_flags,
    stackoverflow_nwp=so_nwp_flags,
    stackoverflow_lr=so_lr_flags,
)

TASK_FLAG_PREFIXES = collections.OrderedDict(
    cifar100="cifar100",
    emnist_cr="emnist_cr",
    emnist_ae="emnist_ae",
    shakespeare="shakespeare",
    stackoverflow_nwp="so_nwp",
    stackoverflow_lr="so_lr",
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
