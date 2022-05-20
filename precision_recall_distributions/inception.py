#!/usr/bin/env python3
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from inception_network import get_inception_features


def embed_images_in_inception(imgs, inception_path, layer_name, batch_size=32):
    input_tensor = tf.placeholder(tf.float32, [None, None, None, 3])

    if not os.path.exists(inception_path):
        raise ValueError('Inception network file not found: ' + inception_path)
    graph = tf.contrib.gan.eval.get_graph_def_from_disk(inception_path)
    feature_tensor = get_inception_features(input_tensor, graph, layer_name)

    embeddings = []
    i = 0
    with tf.Session() as sess:
        while i < len(imgs):
            embeddings.append(
                sess.run(feature_tensor,
                         feed_dict={input_tensor: imgs[i:i + batch_size]}))
            i += batch_size
    return np.concatenate(embeddings, axis=0)
