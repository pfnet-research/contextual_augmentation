#!/usr/bin/env python

from __future__ import print_function

import argparse
import json
import os

from text_classification import text_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-data', default='imdb.binary',
                        choices=['dbpedia', 'imdb.binary', 'imdb.fine',
                                 'TREC', 'stsa.binary', 'stsa.fine',
                                 'custrev', 'mpqa', 'rt-polarity', 'subj'],
                        help='Name of dataset.')
    args = parser.parse_args()
    construct(args)


def construct(args):
    vocab = None

    # Load a dataset
    if args.dataset == 'dbpedia':
        train, test, vocab = text_datasets.get_dbpedia(
            vocab=vocab)
    elif args.dataset.startswith('imdb.'):
        train, test, vocab = text_datasets.get_imdb(
            fine_grained=args.dataset.endswith('.fine'),
            vocab=vocab)
    elif args.dataset in ['TREC', 'stsa.binary', 'stsa.fine',
                          'custrev', 'mpqa', 'rt-polarity', 'subj']:
        train, test, vocab = text_datasets.get_other_text_dataset(
            args.dataset, vocab=vocab)

    if not os.path.exists('vocabs'):
        os.mkdir('vocabs')
    vocab_path = os.path.join('vocabs/{}.vocab.json'.format(args.dataset))
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)


if __name__ == '__main__':
    main()
