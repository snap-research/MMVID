# coding=utf-8
# Taken from https://github.com/google/compare_gan/blob/master/compare_gan/src/fid_score.py
# Copyright 2018 Google LLC & Hwalsuk Lee.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def preprocess_for_inception(images):
    """Preprocess images for inception.

  Args:
    images: images minibatch. Shape [batch size, width, height,
      channels]. Values are in [0..255].

  Returns:
    preprocessed_images
  """

    # Images should have 3 channels.
    assert images.shape[3].value == 3

    # tf.contrib.gan.eval.preprocess_image function takes values in [0, 255]
    with tf.control_dependencies([
            tf.assert_greater_equal(images, 0.0),
            tf.assert_less_equal(images, 255.0)
    ]):
        images = tf.identity(images)

    preprocessed_images = tf.map_fn(fn=tf.contrib.gan.eval.preprocess_image,
                                    elems=images,
                                    back_prop=False)

    return preprocessed_images


def get_inception_features(inputs, inception_graph, layer_name):
    """Compose the preprocess_for_inception function with TFGAN run_inception."""

    preprocessed = preprocess_for_inception(inputs)
    return tf.contrib.gan.eval.run_inception(preprocessed,
                                             graph_def=inception_graph,
                                             output_tensor=layer_name)
