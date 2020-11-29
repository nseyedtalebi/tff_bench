import abc

from absl import flags
import tensorflow_federated as tff

from grfu.utils.utils_impl import lookup_flag_values
from grfu.optimization import optimizer_utils
from define_flags import define_optimizer_flags, define_shared_flags

class Experiment(abc.ABC):

    def __init__(self):
        self.shared_flags = lookup_flag_values(shared_flags)
        self.optimizer_flags = lookup_flag_values(optimizer_flags)
        combined = {**shared_flags,**optimizer_flags}
        for key in combined:
            setattr(self, key, combined[key])

    def get_iterative_process(self):
        return tff.learning.build_federated_averaging_process(
            model_fn=self.model_fn,
            client_optimizer_fn=self.client_optimizer_fn(),
            server_optimizer_fn=self.server_optimizer_fn(),
            client_weight_fn=self.client_weight_fn,
            broadcast_process=self.broadcast_process,
            aggregation_process=self.aggregation_process,
            model_update_aggregation_factory=self.model_update_aggregation_factory,
            use_experimental_simulation_loop=False
        )

    @property
    def model_update_aggregation_factory(self):
        return None

    @property
    def aggregation_process(self):
        return None

    @property
    def broadcast_process(self):
        return None

    @property
    def server_optimzer_fn(self):
        return optimizer_utils.create_optimizer_fn_from_flags('server')

    @property
    def client_optimizer_fn(self):
        return optimizer_utils.create_optimizer_fn_from_flags('client')

    @property
    def client_weight_fn(self):
        return None

    @property
    @abc.abstractmethod
    def model_fn(self):
        pass

    @property
    @abc.abstractmethod
    def training_dataset(self):
        pass

    @property
    @abc.abstractmethod
    def test_dataset(self):
        pass

    @staticmethod
    def experiment_specific_flags():
        return None

    @staticmethod
    def define_experiment_flags():
        define_shared_flags()
        define_optimizer_flags()
        Experiment.experiment_specific_flags()

class DifferentialPrivacyMixin:
    def __init__(self):
        dp_flags = lookup_flag_values(dp_flags)
        for key in dp_flags:
            setattr(self, key, dp_flags[key])

    @property
    def aggregation_process(self):
        dp_query = tff.utils.build_dp_query(
            clip=self.clip,
            noise_multiplier=self.noise_multiplier,
            expected_total_weight=self.clients_per_round,
            adaptive_clip_learning_rate=self.adaptive_clip_learning_rate,
            target_unclipped_quantile=self.target_unclipped_quantile,
            clipped_count_budget_allocation=self.clipped_count_budget_allocation,
            expected_clients_per_round=self.clients_per_round,
        )
        weights_type = tff.learning.framework.weights_type_from_model(self.model_fn)
        return tff.utils.build_dp_aggregate_process(weights_type.trainable, dp_query)
