import typing
from typing import Tuple, Callable

from absl import flags

import grfu.utils.util_impl as util_impl
from grfu.utils.datasets import emnist_dataset
from grfu.utils.models import emnist_models
from experiment import Experiment, DifferentialPrivacyMixin

from grfu.utils.utils_impl import lookup_flag_values
from grfu.optimization import optimizer_utils
from define_flags import define_optimizer_flags, define_shared_flags

import tensorflow_federated as tff
import tensorflow as tf

define_shared_flags()
define_optimizer_flags()
FLAGS = flags.FLAGS
'''
###Example call to training_loop.run
```training_loop.run(
      iterative_process=,
      client_datasets_fn=,
      validation_fn=,
      test_fn=,
      total_rounds=,
      experiment_name=,
      root_output_dir=root_output_dir,
      **kwargs)
```
tff.learning.from_keras_model(
    keras_model,
    loss,
    input_spec,
    loss_weights,
    metrics
)
keras_model: tf.keras.Model,
loss:  tf.keras.losses.Loss,
input_spec,#A structure of tf.TensorSpecs or tff.Type specifying the type of arguments the model expects.
loss_weights: Optional[List[float]] = None,
metrics: Optional[List[tf.keras.metrics.Metric]] = None,
'''

def build_federated_process(
    model_fn,
    process_type="fedavg",#fedavg or fedsgd
    client_weight_fn=1.0
) -> tff.templates.IterativeProcess:
    if process_type == "fedavg":
        process_builder_fn = tff.learning.build_federated_averaging_process
    elif process_type = "fedsgd"
        process_builder_fn = tff.learning.build_federated_sgd_process
    else
        raise ArgumentError(f"Unknown process_type {process_type}. process_type must be 'fedavg' or 'fedsgd'")

    server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
    client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')

    if FLAGS.noise_multiplier:
        if not FLAGS.uniform_weighting:
            raise ValueError('Differential privacy is only implemented for uniform weighting.')
        dp_flags =  util_impl.lookup_flag_values("dp_flags")
        aggregation_process = build_dp_aggregate_process(model_fn, **dp_flags)
    else:
        aggregation_process = None

    #add stuff to set these to include differnet things
    #model_update_aggregation_factory=None,
    #broadcast_process=None,

    return process_builder_fn(
        model_fn,
        client_optimizer_fn,
        server_optimizer_fn,
        client_weight_fn,
        broadcast_process=None,
        aggregation_process=aggregation_process,
        model_update_aggregation_factory=None,
        use_experimental_simulation_loop=False
    )

def build_dp_aggregate_process(model_fn
    clip = 0.05,
    noise_multiplier = 1.1,#From an example for tensorflow_privacy
    adaptive_clip_learning_rate = 0,
    target_unclipped_quantile = 0.5,
    clipped_count_budget_allocation = 0.1
) -> tff.templates.MeasuredProcess:
    dp_query = tff.utils.build_dp_query(
        clip,
        noise_multiplier,
        adaptive_clip_learning_rate,
        target_unclipped_quantile,
        clipped_count_budget_allocation
    )
    weights_type = tff.learning.framework.weights_type_from_model(model_fn)
    return tff.utils.build_dp_aggregate_process(weights_type.trainable, dp_query)

def get_client_datasets_and_eval() -> Tuple[Callable, Callable, Callable]:
    #TODO: impl get standard dataset and coresponding model from flags
    #get model, loss, metrics builds fro client_datasets_fn and evaluate_fn
    #add stuff to main to handle other flags
    train_data, test_data = build_experiment.get_standard_data("some_dataset")

    client_datasets_fn = training_utils.build_client_datasets_fn(
        dataset=train_data,
        clients_per_round=clients_per_round,
        random_seed=client_datasets_random_seed,
    )

    evaluate_fn = training_utils.build_centralized_evaluate_fn(
        eval_dataset=test_data,
        model_builder=model_builder,
        loss_builder=loss_builder,
        metrics_builder=metrics_builder,
    )
    validation_fn = evaluate_fn
    test_fn = evaluate_fn
    return client_datasets_fn, validation_fn, test_fn
