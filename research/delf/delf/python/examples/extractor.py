# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
"""Module to construct DELF feature extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import tensorflow as tf

from delf import feature_extractor

# Minimum dimensions below which DELF features are not extracted (empty
# features are returned). This applies after any resizing is performed.
_MIN_HEIGHT = 10
_MIN_WIDTH = 10


def ResizeImage(image, config):
  """Resizes image according to config.

  Args:
    image: Uint8 array with shape (height, width, 3).
    config: DelfConfig proto containing the model configuration.

  Returns:
    resized_image: Uint8 array with resized image.
    scale_factor: Float with factor used for resizing (If upscaling, larger than
      1; if downscaling, smaller than 1).

  Raises:
    ValueError: If `image` has incorrect number of dimensions/channels.
  """
  if image.ndim != 3:
    raise ValueError('image has incorrect number of dimensions: %d' %
                     image.ndims)
  height, width, channels = image.shape

  if channels != 3:
    raise ValueError('image has incorrect number of channels: %d' % channels)

  if config.max_image_size != -1 and (width > config.max_image_size or
                                      height > config.max_image_size):
    scale_factor = config.max_image_size / max(width, height)
  elif config.min_image_size != -1 and (width < config.min_image_size and
                                        height < config.min_image_size):
    scale_factor = config.min_image_size / max(width, height)
  else:
    # No resizing needed, early return.
    return image, 1.0

  new_shape = (int(width * scale_factor), int(height * scale_factor))
  pil_image = Image.fromarray(image)
  resized_image = np.array(pil_image.resize(new_shape, resample=Image.BILINEAR))

  return resized_image, scale_factor


def MakeExtractor(sess, config, import_scope=None):
  """Creates a function to extract features from an image.

  Args:
    sess: TensorFlow session to use.
    config: DelfConfig proto containing the model configuration.
    import_scope: Optional scope to use for model.

  Returns:
    Function that receives an image and returns features.
  """
  images = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3), name='input')
  reg_feat = feature_extractor.BuildRegModel(images, attentive=False, normalized_image=False)
  restorer = tf.compat.v1.train.Saver(tf.global_variables())
  restorer.restore(sess, config.model_path + 'variables/variables')

  def ExtractorFn(image):
    """Receives an image and returns DELF features.

    If image is too small, returns empty set of features.

    Args:
      image: Uint8 array with shape (height, width, 3) containing the RGB image.

    Returns:
      Tuple (locations, descriptors, feature_scales, attention)
    """
    resized_image, scale_factor = ResizeImage(image, config)

    # If the image is too small, returns empty features.
    if resized_image.shape[0] < _MIN_HEIGHT or resized_image.shape[
        1] < _MIN_WIDTH:
      return np.array([]), np.array([]), np.array([]), np.array([])

    reg_feat_out = sess.run(reg_feat, feed_dict={'input:0': np.expand_dims(resized_image, axis=0)})
    return reg_feat_out

  return ExtractorFn
