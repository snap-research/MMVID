# coding=utf-8
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import cv2
import hashlib

import numpy as np
import inception
import prd_score as prd

parser = argparse.ArgumentParser(
    description='Assessing Generative Models via Precision and Recall',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--reference_dir',
                    type=str,
                    required=True,
                    help='directory containing reference images')
parser.add_argument('--eval_dirs',
                    type=str,
                    nargs='+',
                    required=True,
                    help='directory or directories containing images to be '
                    'evaluated')
parser.add_argument('--eval_labels',
                    type=str,
                    nargs='+',
                    required=True,
                    help='labels for the eval_dirs (must have same size)')
parser.add_argument('--num_clusters',
                    type=int,
                    default=20,
                    help='number of cluster centers to fit')
parser.add_argument('--num_angles',
                    type=int,
                    default=1001,
                    help='number of angles for which to compute PRD, must be '
                    'in [3, 1e6]')
parser.add_argument(
    '--num_runs',
    type=int,
    default=10,
    help='number of independent runs over which to average the '
    'PRD data')
parser.add_argument('--plot_path',
                    type=str,
                    default=None,
                    help='path for final plot file (can be .png or .pdf)')
parser.add_argument('--cache_dir',
                    type=str,
                    default='/tmp/prd_cache/',
                    help='cache directory')
parser.add_argument('--inception_path',
                    type=str,
                    default='/tmp/prd_cache/inception.pb',
                    help='path to pre-trained Inception.pb file')
parser.add_argument('--silent',
                    dest='verbose',
                    action='store_false',
                    help='disable logging output')

args = parser.parse_args()


def generate_inception_embedding(imgs, inception_path, layer_name='pool_3:0'):
    return inception.embed_images_in_inception(imgs, inception_path,
                                               layer_name)


def load_or_generate_inception_embedding(directory, cache_dir, inception_path):
    hash = hashlib.md5(directory.encode('utf-8')).hexdigest()
    path = os.path.join(cache_dir, hash + '.npy')
    if os.path.exists(path):
        embeddings = np.load(path)
        return embeddings
    imgs = load_images_from_dir(directory)
    embeddings = generate_inception_embedding(imgs, inception_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(path, 'wb') as f:
        np.save(f, embeddings)
    return embeddings


def load_images_from_dir(directory, types=('png', 'jpg', 'bmp', 'gif')):
    paths = [
        os.path.join(directory, fn) for fn in os.listdir(directory)
        if os.path.splitext(fn)[-1][1:] in types
    ]
    # images are in [0, 255]
    imgs = [
        cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        for path in paths
    ]
    return np.array(imgs)


if __name__ == '__main__':
    if len(args.eval_dirs) != len(args.eval_labels):
        raise ValueError(
            'Number of --eval_dirs must be equal to number of --eval_labels.')

    reference_dir = os.path.abspath(args.reference_dir)
    eval_dirs = [os.path.abspath(directory) for directory in args.eval_dirs]

    if args.verbose:
        print('computing inception embeddings for ' + reference_dir)
    real_embeddings = load_or_generate_inception_embedding(
        reference_dir, args.cache_dir, args.inception_path)
    prd_data = []
    for directory in eval_dirs:
        if args.verbose:
            print('computing inception embeddings for ' + directory)
        eval_embeddings = load_or_generate_inception_embedding(
            directory, args.cache_dir, args.inception_path)
        if args.verbose:
            print('computing PRD')
        prd_data.append(
            prd.compute_prd_from_embedding(eval_data=eval_embeddings,
                                           ref_data=real_embeddings,
                                           num_clusters=args.num_clusters,
                                           num_angles=args.num_angles,
                                           num_runs=args.num_runs))
    if args.verbose:
        print('plotting results')

    print()
    f_beta_data = [
        prd.prd_to_max_f_beta_pair(precision, recall, beta=8)
        for precision, recall in prd_data
    ]
    print('F_8   F_1/8     model')
    for directory, f_beta in zip(eval_dirs, f_beta_data):
        print('%.3f %.3f     %s' % (f_beta[0], f_beta[1], directory))

    prd.plot(prd_data, labels=args.eval_labels, out_path=args.plot_path)
