#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
import collections
import io
import json
import os

import numpy as np
import progressbar

import chainer
from chainer import cuda
from chainer.dataset import convert


class UnkDropout(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, unk, ratio=0.01):
        self.dataset = dataset
        self.unk = unk
        self.ratio = ratio

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        x, y = self.dataset[i]
        if chainer.config.train:
            rand = np.random.rand(x.size - 2) < self.ratio
            # keep bos and eos
            _x = x.copy()
            _x[1:-1] = np.where(rand, self.unk, _x[1:-1])
            return (_x, y)
        return (x, y)


def convert_xt_batch_seq(xt_batch_seq, gpu):
    batchsize = len(xt_batch_seq[0])
    seq_len = len(xt_batch_seq)
    xt_batch_seq = np.array(xt_batch_seq, 'i')
    # (bproplen, batch, 2)
    xt_batch_seq = convert.to_device(gpu, xt_batch_seq)
    xp = cuda.get_array_module(xt_batch_seq)
    x_seq_batch = xp.split(
        xt_batch_seq[:, :, 0].T.reshape(batchsize * seq_len),
        batchsize, axis=0)
    t_seq_batch = xp.split(
        xt_batch_seq[:, :, 1].T.reshape(batchsize * seq_len),
        batchsize, axis=0)
    return x_seq_batch, t_seq_batch


def count_words_from_file(counts, file_path):
    bar = progressbar.ProgressBar()
    for l in bar(io.open(file_path, encoding='utf-8')):
        # TODO: parallel
        if l.strip():
            words = l.strip().split()
            for word in words:
                counts[word] += 1
    return counts


def count_words(dataset, alpha=0.4):
    counts = collections.defaultdict(int)
    for w in dataset:
        counts[w] += 1
    counts = [counts[i] for i in range(len(counts))]
    counts = np.array(counts, 'f')
    counts /= counts.sum()
    counts = counts ** alpha
    counts = counts.tolist()
    return counts


def make_chain_dataset(file_path, vocab={}, update_vocab=False,
                       chain_length=2):
    dataset = []
    chain = []
    unk_id = vocab['<unk>']

    def make_array(chain):
        array_chain = []
        for words in chain:
            tokens = []
            for word in words:
                if update_vocab:
                    if word not in vocab:
                        vocab[word] = len(vocab)
                tokens.append(vocab.get(word, unk_id))
            array_chain.append(np.array(tokens, 'i'))
        return array_chain

    for line in io.open(file_path, encoding='utf-8'):
        if not line.strip():
            if len(chain) >= chain_length:
                dataset.append(make_array(chain))
                chain = []
            continue
        words = line.strip().split() + ['<eos>']
        chain.append(words)
    if len(chain) >= chain_length:
        dataset.append(make_array(chain))
    return dataset, vocab


def tokenize_text(file_path, vocab={}, update_vocab=False):
    tokens = []
    unk_id = vocab['<unk>']
    with io.open(file_path, encoding='utf-8') as f:
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                if update_vocab:
                    if word not in vocab:
                        vocab[word] = len(vocab)
                tokens.append(vocab.get(word, unk_id))
    return tokens, vocab


def get_wikitext_words_and_vocab(
        name='wikitext-2', base_dir='datasets', vocab=None):
    assert(name in ['wikitext-2', 'wikitext-103'])
    base_dir2 = os.path.join(base_dir, name)
    predata_path = os.path.join(base_dir2, 'preprocessed_data.json')
    if os.path.exists(predata_path) and vocab is None:
        train, valid, test, vocab = json.load(open(predata_path))
    else:
        prepared_vocab = (vocab is not None)
        if not prepared_vocab:
            vocab = {'<eos>': 0, '<unk>': 1}
        train, vocab = tokenize_text(
            os.path.join(base_dir2, 'wiki.train.tokens'),
            vocab, update_vocab=not prepared_vocab)
        valid, _ = tokenize_text(
            os.path.join(base_dir2, 'wiki.valid.tokens'),
            vocab, update_vocab=False)
        test, _ = tokenize_text(
            os.path.join(base_dir2, 'wiki.test.tokens'),
            vocab, update_vocab=False)
        json.dump([train, valid, test, vocab], open(predata_path, 'w'))
    return train, valid, test, vocab
