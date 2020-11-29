from absl import flags

from grfu.utils.datasets import emnist_dataset
from experiment import Experiment, DifferentialPrivacyMixin

class emnistExperiment(Experiment):

    def __init__(self):
        super(Experiment)
        super(DifferentialPrivacyMixin)

    @property
    def model_fn(self):
        pass

    @property
    def training_dataset(self):
        return emnist_dataset.get_emnist_datasets(
        client_batch_size,
        client_epochs_per_round,
        max_batches_per_client=max_batches_per_client,
        only_digits=False)

    @property
    def test_dataset(self):
        return emnist_dataset.get_centralized_datasets(
        train_batch_size=client_batch_size,
        max_test_batches=max_eval_batches,
        only_digits=False)

    @staticmethod
    def experiment_specific_flags():
        flags.DEFINE_enum("model", "cnn", ["cnn","2nn"],
        """A string specifying the model used for character recognition.
      Can be one of `cnn` and `2nn`, corresponding to a CNN model and a densely
      connected 2-layer model (respectively).""")
