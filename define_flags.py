from absl import flags

from grfu.utils import utils_impl
from grfu.optimization import fed_avg_schedule
from grfu.optimization import optimizer_utils


def define_dp_flags():
    with utils_impl.record_hparam_flags() as dp_flags:
        # Differential privacy flags
        flags.DEFINE_float("clip", 0.05, "Initial clip.")
        flags.DEFINE_float(
            "noise_multiplier", None, "Noise multiplier. If None, no DP is used."
        )
        flags.DEFINE_float(
            "adaptive_clip_learning_rate", 0, "Adaptive clip learning rate."
        )
        flags.DEFINE_float(
            "target_unclipped_quantile", 0.5, "Target unclipped quantile."
        )
        flags.DEFINE_float(
            "clipped_count_budget_allocation",
            0.1,
            "Fraction of privacy budget to allocate for clipped counts.",
        )


def define_optimizer_flags():
    with utils_impl.record_hparam_flags() as optimizer_flags:
        # Defining optimizer flags
        optimizer_utils.define_optimizer_flags("client")
        optimizer_utils.define_optimizer_flags("server")
        optimizer_utils.define_lr_schedule_flags("client")
        optimizer_utils.define_lr_schedule_flags("server")


def define_shared_flags():
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
    return utils_impl.lookup_flag_values(shared_flags)
