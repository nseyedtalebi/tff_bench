from typing import Optional, Tuple, Dict, String

import tensorflow as tf
import tensorflow_federated as tff


class CentralizedDataset:

	test: tf.Data.Dataset
	train: tf.Data.Dataset
	validate: tf.Data.Dataset
	been_loaded: bool
	train_batch_size: int
	test_batch_size: int
	validation_batch_size: int
	max_train_batches: int
	max_test_batches: int
	max_validation_batches: int
	kwargs: Dict[String,_]
	#todo: add something to read rng seeds in from text file
	#todo: add property for dataset name
	#keep combining dataset files, making sure everything is cached as you go
	CentralizedDataset(train_batch_size: int,
						test_batch_size: Optional[int] = 500,
                        validation_batch_size: Optional[int] = 500,
                        max_train_batches: Optional[int] = None,
                        max_test_batches: Optional[int] = None,
                        max_validation_batches: Optional[int] = None,
                        **kwargs
                       ) -> Tuple[tf.Data.Dataset, tf.Data.Dataset, tf.Data.Dataset]:
		self.been_loaded = False
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.validation_batch_size = validation_batch_size
		self.max_train_batches = max_train_batches
		self.max_test_batches = max_test_batches
		self.max_validation_batches = max_validation_batches

	def load_and_preprocess(self, clientdata: tff.simulation.datasets.ClientData,
		test_preprocess_fn=None, train_preprocess_fn=None, validation_preprocess_fn=None,

		cache_dir= '.'):
		raw_test, raw_train, raw_validate = clientdata.load_data(cache_dir=cache_dir)
		self.test = test_preprocess_fn(raw_test.create_tf_dataset_from_all_clients(seed=None)).cache(cache_dir=cache_dir)
		self.train = raw_train.preprocess(train_preprocess_fn).cache(cache_dir=cache_dir)
		self.validate = None


'''
def get_centralized_datasets(train_batch_size: int,
                             
                            ):
'''