#!/usr/bin/env python
from __future__ import print_function

import argparse
import json


def get_basic_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of layers of RNN or MLP following CNN')
    parser.add_argument('--dropout', '-d', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--dataset', '-data', default='imdb.binary',
                        choices=['dbpedia', 'imdb.binary', 'imdb.fine',
                                 'TREC', 'stsa.binary', 'stsa.fine',
                                 'custrev', 'mpqa', 'rt-polarity', 'subj'],
                        help='Name of dataset.')
    parser.add_argument('--model', '-model', default='cnn',
                        choices=['cnn', 'rnn', 'bow'],
                        help='Name of encoder model type.')

    parser.add_argument('--bilm', '-bilm')
    parser.add_argument('--bilm-unit', '-bilm-u', type=int, default=1024)
    parser.add_argument('--bilm-layer', '-bilm-l', type=int, default=1)
    parser.add_argument('--bilm-dropout', '-bilm-d', type=float, default=0.)

    parser.add_argument('--bilm-ratio', '-bilm-r', type=float, default=0.25)
    parser.add_argument('--bilm-temp', '-bilm-t', type=float, default=1.)
    parser.add_argument('--bilm-mode', '-bilm-m', default='sampling',
                        choices=['weighted_sum', 'sampling'])
    parser.add_argument('--bilm-gumbel', action='store_true')
    parser.add_argument('--bilm-wordwise', action='store_true', default=True)
    parser.add_argument('--bilm-add-original', type=float, default=0.)
    parser.add_argument('--bilm-residual', type=float, default=0.,
                        help='if not 0, (original + context) * THIS')

    parser.add_argument('--resume-vocab')

    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--seed', type=int, default=2018)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--stop-epoch', type=int, default=10)

    parser.add_argument('--no-label', action='store_true')

    return parser


if __name__ == '__main__':
    parser = get_basic_arg_parser()
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))
