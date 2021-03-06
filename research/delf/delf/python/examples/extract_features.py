# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
"""Extracts DELF features from a list of images, saving them to file.

The images must be in JPG format. The program checks if descriptors already
exist, and skips computation for those.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import feature_io
from delf import extractor
import numpy as np

cmd_args = None

# Extension of feature files.
_DELF_EXT = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def _ReadImageList(list_path, prefix):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [os.path.join(prefix, entry.rstrip()) for entry in image_paths]
  return image_paths


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Read list of images.
  tf.logging.info('Reading list of images...')
  image_paths = _ReadImageList(cmd_args.list_images_path, cmd_args.prefix)
  num_images = len(image_paths)
  tf.logging.info('done! Found %d images', num_images)

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.gfile.FastGFile(cmd_args.config_path, 'r') as f:
    text_format.Merge(f.read(), config)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Reading list of images.
    filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    image_tf = tf.image.resize(image_tf, (448, 448))

    with tf.Session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)

      extractor_fn = extractor.MakeExtractor(sess, config)

      # Start input enqueue threads.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      start = time.clock()
      for i in range(num_images):
        # Write to log-info once in a while.
        if i == 0:
          tf.logging.info('Starting to extract DELF features from images...')
        elif i % _STATUS_CHECK_ITERATIONS == 0:
          elapsed = (time.clock() - start)
          tf.logging.info(
              'Processing image %d out of %d, last %d '
              'images took %f seconds', i, num_images, _STATUS_CHECK_ITERATIONS,
              elapsed)
          start = time.clock()

        # # Get next image.
        im = sess.run(image_tf)

        # If descriptor already exists, skip its computation.
        save_path = image_paths[i]
        save_path = save_path.replace('undist_images', 'reg_feat')
        save_path = save_path.replace('.jpg', '.bin')
        if cmd_args.skip_extracted and tf.gfile.Exists(save_path):
          tf.logging.info('Skipping %s', image_paths[i])
          continue

        # Extract and save features.
        reg_feat_out = extractor_fn(im)
        reg_feat_out = np.squeeze(reg_feat_out, axis=0)

        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        reg_feat_out.astype(np.float32).tofile(save_path)

      # Finalize enqueue threads.
      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--config_path',
      type=str,
      default='delf_config_example.pbtxt',
      help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
  parser.add_argument(
      '--list_images_path',
      type=str,
      default='list_images.txt',
      help="""
      Path to list of images whose DELF features will be extracted.
      """)
  parser.add_argument(
      '--prefix',
      type=str,
      default='/data/GL3D',
      help="""
      Prefix of input images.
      """)
  parser.add_argument(
      '--skip_extracted',
      type=bool,
      default=False,
      help="""
      Whether to skip extract features.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
