import tensorflow as tf


class CalculateDPEpsilon(tf.keras.metrics.Sum):
    def __init__(
        self,
        num_examples_all_clients,
        epochs,
        name: str = "dp_epsilon",
        dtype=tf.float64,
    ):  # pylint: disable=useless-super-delegation
        self.num_examples_all_clients = num_examples_all_clients
        self.epochs = epochs
        super().__init__(name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        num_examples = tf.shape(y_pred)[0]
        q = num_examples / self.num_examples_all_clients
        steps = int(
            math.ceil(self.epochs * self.num_examples_all_clients / num_examples)
        )
        orders = (
            [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
            + list(range(5, 64))
            + [128, 256, 512]
        )
        rdp = compute_rdp(q, noise_multiplier, steps, orders)
        eps, _, opt_order = get_privacy_spent(
            orders, rdp, target_eps=None, target_delta=1 / num_examples
        )
        return super().update_state(eps)


'''
can subclass a sum metric to get epsilon
can calculate epsilon after the fact using metrics output
def get_dp_metric():
    q = batch_size / n  # q - the sampling ratio.
    if q > 1:
        raise app.UsageError('n must be larger than the batch size.')
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                list(range(5, 64)) + [128, 256, 512])
    steps = int(math.ceil(epochs * n / batch_size))

    def compute_rdp(q, noise_multiplier, steps, orders):
  """Compute RDP of the Sampled Gaussian Mechanism.
  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
        to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders, can be np.inf.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp(q, noise_multiplier, orders)
  else:
    rdp = np.array([_compute_rdp(q, noise_multiplier, order)
                    for order in orders])

  return rdp * steps


def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  """Compute delta (or eps) for given eps (or delta) from RDP values.
  Args:
    orders: An array (or a scalar) of RDP orders.
    rdp: An array of RDP values. Must be of the same length as the orders list.
    target_eps: If not None, the epsilon for which we compute the corresponding
              delta.
    target_delta: If not None, the delta for which we compute the corresponding
              epsilon. Exactly one of target_eps and target_delta must be None.
  Returns:
    eps, delta, opt_order.
  Raises:
    ValueError: If target_eps and target_delta are messed up.
  """

  return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)
 '''
