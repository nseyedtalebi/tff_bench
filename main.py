"""
everything here should help to prepare a training loop and run it like in the example scripts
example call:
  training_loop.run(
      iterative_process=,
      client_datasets_fn=,
      validation_fn=,
      test_fn=,
      total_rounds=,
      experiment_name=,
      root_output_dir=root_output_dir,
      **kwargs)

###iterative_process:

- ```
    tff.learning.build_federated_averaging_process(
    model_fn: Callable[[], tff.learning.Model],
    client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer] = DEFAULT_SERVER_OPTIMIZER_FN,
    client_weight_fn: Callable[[Any], tf.Tensor] = None,
    *,
    broadcast_process: Optional[tff.templates.MeasuredProcess] = None,
    aggregation_process: Optional[tff.templates.MeasuredProcess] = None,
    model_update_aggregation_factory: Optional[tff.aggregators.AggregationProcessFactory] = None,
    use_experimental_simulation_loop: bool = False
) -> tff.templates.IterativeProcess
```

- ```tff.learning.build_federated_sgd_process(
    model_fn: Callable[[], tff.learning.Model],
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer] = DEFAULT_SERVER_OPTIMIZER_FN,
    client_weight_fn: Callable[[Any], tf.Tensor] = None,
    *,
    broadcast_process: Optional[tff.templates.MeasuredProcess] = None,
    aggregation_process: Optional[tff.templates.MeasuredProcess] = None,
    model_update_aggregation_factory: Optional[tff.aggregators.AggregationProcessFactory] = None,
    use_experimental_simulation_loop: bool = False
) -> tff.templates.IterativeProcess
```

###client_datsets
- ```google_research.federated.utils.training_utils.build_client_datasets_fn(
    dataset: tff.simulation.ClientData,
    clients_per_round: int,
    random_seed: Optional[int] = None
) -> Callable[[int], List[tf.data.Dataset]]:
```

###validation_fn

- ```
training_utils.build_centralized_evaluate_fn(
      eval_dataset=emnist_test,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)
```

- ```
google_research.federated.utils.training_utils.build_federated_evaluate_fn(
    eval_dataset: tff.simulation.ClientData,
    model_builder: Callable[[], tf.keras.Model],
    metrics_builder: Callable[[], List[tf.keras.metrics.Metric]],
    clients_per_round: int,
    random_seed: Optional[int] = None,
    quantiles: Optional[Iterable[float]] = DEFAULT_QUANTILES,
) -> Callable[[tff.learning.ModelWeights, int], Dict[str, Any]]:
```

"""
