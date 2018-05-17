# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Library to upload benchmark generated by BenchmarkLogger to remote repo.

This library require google cloud bigquery lib as dependency, which can be
installed with:
  > pip install --upgrade google-cloud-bigquery
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from google.cloud import bigquery

import tensorflow as tf


class BigQueryUploader(object):
  """Upload the benchmark and metric info from JSON input to BigQuery. """

  def __init__(self, gcp_project=None, credentials=None):
    """Initialized BigQueryUploader with proper setting.

    Args:
      gcp_project: string, the name of the GCP project that the log will be
        uploaded to. The default project name will be detected from local
        environment if no value is provided.
      credentials: google.auth.credentials. The credential to access the
        BigQuery service. The default service account credential will be
        detected from local environment if no value is provided. Please use
        google.oauth2.service_account.Credentials to load credential from local
        file for the case that the test is run out side of GCP.
    """
    self._bq_client = bigquery.Client(
        project=gcp_project, credentials=credentials)

  def upload_benchmark_run_json(
      self, dataset_name, table_name, run_id, run_json):
    """Upload benchmark run information to Bigquery.

    Args:
      dataset_name: string, the name of bigquery dataset where the data will be
        uploaded.
      table_name: string, the name of bigquery table under the dataset where
        the data will be uploaded.
      run_id: string, a unique ID that will be attached to the data, usually
        this is a UUID4 format.
      run_json: dict, the JSON data that contains the benchmark run info.
    """
    run_json["model_id"] = run_id
    self._upload_json(dataset_name, table_name, [run_json])

  def upload_benchmark_metric_json(
      self, dataset_name, table_name, run_id, metric_json_list):
    """Upload metric information to Bigquery.

    Args:
      dataset_name: string, the name of bigquery dataset where the data will be
        uploaded.
      table_name: string, the name of bigquery table under the dataset where
        the metric data will be uploaded. This is different from the
        benchmark_run table.
      run_id: string, a unique ID that will be attached to the data, usually
        this is a UUID4 format. This should be the same as the benchmark run_id.
      metric_json_list: list, a list of JSON object that record the metric info.
    """
    for m in metric_json_list:
      m["run_id"] = run_id
    self._upload_json(dataset_name, table_name, metric_json_list)

  def upload_benchmark_run_file(
      self, dataset_name, table_name, run_id, run_json_file):
    """Upload benchmark run information to Bigquery from input json file.

    Args:
      dataset_name: string, the name of bigquery dataset where the data will be
        uploaded.
      table_name: string, the name of bigquery table under the dataset where
        the data will be uploaded.
      run_id: string, a unique ID that will be attached to the data, usually
        this is a UUID4 format.
      run_json_file: string, the file path that contains the run JSON data.
    """
    with tf.gfile.GFile(run_json_file) as f:
      benchmark_json = json.load(f)
      self.upload_benchmark_run_json(
          dataset_name, table_name, run_id, benchmark_json)

  def upload_metric_file(
      self, dataset_name, table_name, run_id, metric_json_file):
    """Upload metric information to Bigquery from input json file.

    Args:
      dataset_name: string, the name of bigquery dataset where the data will be
        uploaded.
      table_name: string, the name of bigquery table under the dataset where
        the metric data will be uploaded. This is different from the
        benchmark_run table.
      run_id: string, a unique ID that will be attached to the data, usually
        this is a UUID4 format. This should be the same as the benchmark run_id.
      metric_json_file: string, the file path that contains the metric JSON
        data.
    """
    with tf.gfile.GFile(metric_json_file) as f:
      metrics = []
      for line in f:
        metrics.append(json.loads(line.strip()))
      self.upload_benchmark_metric_json(
          dataset_name, table_name, run_id, metrics)

  def _upload_json(self, dataset_name, table_name, json_list):
    # Find the unique table reference based on dataset and table name, so that
    # the data can be inserted to it.
    table_ref = self._bq_client.dataset(dataset_name).table(table_name)
    errors = self._bq_client.insert_rows_json(table_ref, json_list)
    if errors:
      tf.logging.error(
          "Failed to upload benchmark info to bigquery: {}".format(errors))