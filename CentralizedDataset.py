from typing import Optional, Tuple, Dict, String

import tensorflow as tf
import tensorflow_federated as tff

cache_dir = 'C:\\Windows\\Temp'
def get_builtin_dataset(name: str,
						train_batch_size: int,
						test_batch_size: Optional[int] = 500,
                        validation_batch_size: Optional[int] = 500,
                        max_train_batches: Optional[int] = None,
                        max_test_batches: Optional[int] = None,
                        max_validation_batches: Optional[int] = None
                       ) -> Tuple[tf.Data.Dataset, tf.Data.Dataset, tf.Data.Dataset]:
	
	if name == 'stackoverflow':
		
	if name == 'stackoverflow_lr':
		pass
	if name == 'cifar100':
		pass
	if name == 'emnist':
		pass
	if name == 'emnist_ae':
		pass
	if name == 'shakespeare':
		pass

def load_and_preprocess(clientdata: tff.simulation.datasets.ClientData,
						test_preprocess_fn=None,
						train_preprocess_fn=None,
						validation_preprocess_fn=None,
					    cache_dir= '.'):
	raw_test, raw_train, raw_validate = clientdata.load_data(cache_dir=cache_dir)
	test = test_preprocess_fn(raw_test.create_tf_dataset_from_all_clients(seed=None)).cache(cache_dir=cache_dir)
	train = raw_train.preprocess(train_preprocess_fn).cache(cache_dir=cache_dir)
	validate = None
	return test, train, validate
