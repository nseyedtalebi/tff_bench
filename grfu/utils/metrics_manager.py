# Copyright 2019, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility class for saving and loading scalar experiment metrics."""

import collections
import csv
import os.path
import shutil
import tempfile
from typing import Any, Dict, List, Tuple, Sequence, Set, Union

from absl import logging
import numpy as np
import tensorflow as tf
import tree

_QUOTING = csv.QUOTE_NONNUMERIC


def _create_if_not_exists(path):
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def _read_from_csv(
    file_name: str) -> Tuple[Sequence[str], List[Dict[str, Any]]]:
  """Returns a list of fieldnames and a list of metrics from a given CSV."""
  with tf.io.gfile.GFile(file_name, 'r') as csv_file:
    reader = csv.DictReader(csv_file, quoting=_QUOTING)
    fieldnames = reader.fieldnames
    csv_metrics = list(reader)
  return fieldnames, csv_metrics


def _write_to_csv(metrics: List[Dict[str, Any]], file_name: str,
                  fieldnames: Union[Sequence[str], Set[str]]):
  """Writes a list of metrics to CSV in an atomic fashion."""
  tmp_dir = tempfile.mkdtemp(prefix='atomic_write_to_csv_tmp')
  tmp_name = os.path.join(tmp_dir, os.path.basename(file_name))
  assert not tf.io.gfile.exists(tmp_name), 'File [{!s}] already exists'.format(
      tmp_name)

  # Write to a temporary GFile.
  with tf.io.gfile.GFile(tmp_name, 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=_QUOTING)
    writer.writeheader()
    for metric_row in metrics:
      writer.writerow(metric_row)

  # Copy to a temporary GFile next to the target, allowing for an atomic move.
  tmp_gfile_name = os.path.join(
      os.path.dirname(file_name), '{}.tmp{}'.format(
          os.path.basename(file_name),
          np.random.randint(0, 2**63, dtype=np.int64)))
  tf.io.gfile.copy(src=tmp_name, dst=tmp_gfile_name, overwrite=True)

  # Finally, do an atomic rename and clean up.
  tf.io.gfile.rename(tmp_gfile_name, file_name, overwrite=True)
  shutil.rmtree(tmp_dir)


def _append_to_csv(metrics_to_append: Dict[str, Any], file_name: str):
  """Appends `metrics` to a CSV.

  Here, the CSV is assumed to have first row representing the CSV's fieldnames.
  If `metrics` contains keys not in the CSV fieldnames, the CSV is re-written
  in order using the union of the fieldnames and `metrics.keys`.

  Args:
    metrics_to_append: A dictionary of metrics.
    file_name: The CSV file_name.

  Returns:
    A list of fieldnames for the updated CSV.
  """
  new_fieldnames = metrics_to_append.keys()
  with tf.io.gfile.GFile(file_name, 'a+') as csv_file:
    reader = csv.DictReader(csv_file, quoting=_QUOTING)
    current_fieldnames = reader.fieldnames

    if current_fieldnames is None:
      writer = csv.DictWriter(
          csv_file, fieldnames=list(new_fieldnames), quoting=_QUOTING)
      writer.writeheader()
      writer.writerow(metrics_to_append)
      return new_fieldnames
    elif new_fieldnames <= set(current_fieldnames):
      writer = csv.DictWriter(
          csv_file, fieldnames=current_fieldnames, quoting=_QUOTING)
      writer.writerow(metrics_to_append)
      return current_fieldnames
    else:
      metrics = list(reader)

  expanded_fieldnames = set(current_fieldnames).union(new_fieldnames)
  metrics.append(metrics_to_append)
  _write_to_csv(metrics, file_name, expanded_fieldnames)
  return expanded_fieldnames


class ScalarMetricsManager():
  """Utility class for saving/loading scalar experiment metrics.

  The metrics are backed by CSVs stored on the file system.
  """

  def __init__(self,
               root_metrics_dir: str = '/tmp',
               prefix: str = 'experiment'):
    """Returns an initialized `ScalarMetricsManager`.

    This class will maintain metrics in a CSV file in the filesystem. The path
    of the file is {`root_metrics_dir`}/{`prefix`}.metrics.csv. To use this
    class upon restart of an experiment at an earlier round number, you can
    initialize and then call the clear_rounds_after() method to remove all rows
    for round numbers later than the restart round number. This ensures that no
    duplicate rows of data exist in the CSV.

    Args:
      root_metrics_dir: A path on the filesystem to store CSVs.
      prefix: A string to use as the prefix of file_name. Usually the name of a
        specific run in a larger grid of experiments sharing a common
        `root_metrics_dir`.

    Raises:
      ValueError: If `root_metrics_dir` is empty string.
      ValueError: If `prefix` is empty string.
      ValueError: If the specified metrics csv file already exists but does not
        contain a `round_num` column.
    """
    super().__init__()
    if not root_metrics_dir:
      raise ValueError('Empty string passed for root_metrics_dir argument.')
    if not prefix:
      raise ValueError('Empty string passed for prefix argument.')

    _create_if_not_exists(root_metrics_dir)
    self._metrics_file = os.path.join(root_metrics_dir, f'{prefix}.metrics.csv')
    if not tf.io.gfile.exists(self._metrics_file):
      with tf.io.gfile.GFile(self._metrics_file, 'w') as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=['round_num'], quoting=_QUOTING)
        writer.writeheader()

    current_fieldnames, current_metrics = _read_from_csv(self._metrics_file)

    if current_metrics and 'round_num' not in current_fieldnames:
      raise ValueError(
          f'The specified csv file ({self._metrics_file}) already exists '
          'but was not created by ScalarMetricsManager (it does not contain a '
          '`round_num` column.')

    if not current_metrics:
      self._latest_round_num = None
    else:
      self._latest_round_num = current_metrics[-1]['round_num']

  def update_metrics(self, round_num,
                     metrics_to_append: Dict[str, Any]) -> Dict[str, Any]:
    """Updates the stored metrics data with metrics for a specific round.

    The specified `round_num` must be later than the latest round number for
    which metrics exist in the stored metrics data. This method will atomically
    update the stored CSV file. Also, if stored metrics already exist and
    `metrics_to_append` contains a new, previously unseen metric name, a new
    column in the dataframe will be added for that metric, and all previous rows
    will fill in with NaN values for the metric.

    Args:
      round_num: Communication round at which `metrics_to_append` was collected.
      metrics_to_append: A dictionary of metrics collected during `round_num`.
        These metrics can be in a nested structure, but the nesting will be
        flattened for storage in the CSV (with the new keys equal to the paths
        in the nested structure).

    Returns:
      A `collections.OrderedDict` of the data just added in a new row to the
        pandas.DataFrame. Compared with the input `metrics_to_append`, this data
        is flattened, with the key names equal to the path in the nested
        structure. Also, `round_num` has been added as an additional key.

    Raises:
      ValueError: If the provided round number is negative.
      ValueError: If the provided round number is less than or equal to the
        latest round number in the stored metrics data.
    """
    if round_num < 0:
      raise ValueError(f'Attempting to append metrics for round {round_num}, '
                       'which is negative.')
    if self._latest_round_num and round_num <= self._latest_round_num:
      raise ValueError(f'Attempting to append metrics for round {round_num}, '
                       'but metrics already exist through round '
                       f'{self._latest_round_num}.')

    # Add the round number to the metrics before storing to csv file. This will
    # be used if a restart occurs, to identify which metrics to trim in the
    # _clear_invalid_rounds() method.
    metrics_to_append['round_num'] = round_num

    flat_metrics = tree.flatten_with_path(metrics_to_append)
    flat_metrics = [
        ('/'.join(map(str, path)), item) for path, item in flat_metrics
    ]
    flat_metrics = collections.OrderedDict(flat_metrics)
    _append_to_csv(flat_metrics, self._metrics_file)
    self._latest_round_num = round_num

    return flat_metrics

  def get_metrics(self) -> Tuple[Sequence[str], List[Dict[str, Any]]]:
    """Retrieve the stored experiment metrics data for all rounds.

    Returns:
      A sequence representing all possible keys for the metrics, and a list
      containing experiment metrics data for all rounds. Each entry in the list
      is a dictionary corresponding to a given round. The data has been
      flattened, with the column names equal to the path in the original nested
      metric structure. There is a fieldname `round_num` to indicate the round
      number.
    """
    return _read_from_csv(self._metrics_file)

  def clear_all_rounds(self) -> None:
    """Existing metrics for all rounds are cleared out.

    This method will atomically update the stored CSV file.
    """
    with tf.io.gfile.GFile(self._metrics_file, 'w') as csv_file:
      writer = csv.DictWriter(
          csv_file, fieldnames=['round_num'], quoting=_QUOTING)
      writer.writeheader()
    self._latest_round_num = None

  def clear_rounds_after(self, last_valid_round_num: int) -> None:
    """Metrics for rounds greater than `last_valid_round_num` are cleared out.

    By using this method, this class can be used upon restart of an experiment
    at `last_valid_round_num` to ensure that no duplicate rows of data exist in
    the CSV file. This method will atomically update the stored CSV file.

    Args:
      last_valid_round_num: All metrics for rounds later than this are expunged.

    Raises:
      RuntimeError: If metrics do not exist (none loaded during construction '
        nor recorded via `update_metrics()` and `last_valid_round_num` is not
        zero.
      ValueError: If `last_valid_round_num` is negative.
    """
    if last_valid_round_num < 0:
      raise ValueError('Attempting to clear metrics after round '
                       f'{last_valid_round_num}, which is negative.')
    if self._latest_round_num is None:
      if last_valid_round_num == 0:
        return
      raise RuntimeError('Metrics do not exist yet.')

    reduced_fieldnames = set(['round_num'])
    _, metrics = _read_from_csv(self._metrics_file)
    reduced_metrics = []
    for metric_row in metrics:
      if metric_row['round_num'] <= last_valid_round_num:
        reduced_fieldnames = reduced_fieldnames.union(metric_row.keys())
        reduced_metrics.append(metric_row)

    _write_to_csv(reduced_metrics, self._metrics_file, reduced_fieldnames)
    self._latest_round_num = last_valid_round_num

  @property
  def metrics_filename(self) -> str:
    return self._metrics_file
