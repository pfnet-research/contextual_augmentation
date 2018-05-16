from __future__ import print_function
import argparse
import json

import numpy as np

import chainer
from chainer import cuda
from chainer import training
from chainer.training import extensions
from chainer import serializers

import utils
import chain_utils
import nets


def alter_embedding(model, old_vocab, new_vocab):
    assert('<eos>' in old_vocab)
    assert('<unk>' in old_vocab)
    assert('<eos>' in new_vocab)
    assert('<unk>' in new_vocab)

    old_embed = np.array(model.embed.W.data)
    old_output_W = np.array(model.output.W.data)
    old_output_b = np.array(model.output.b.data)
    delattr(model, 'embed')
    delattr(model, 'output')
    count = 0
    with model.init_scope():
        model.embed = chainer.links.EmbedID(len(new_vocab), old_embed.shape[1],
                                            ignore_label=-1)
        model.output = nets.NormalOutputLayer(
            old_embed.shape[1], len(new_vocab))
        print(model.embed.W.data.shape)
        print(model.output.W.data.shape)

        for w, idx in new_vocab.items():
            if w in old_vocab:
                model.embed.W.data[idx] = old_embed[old_vocab[w]]
                model.output.W.data[idx] = old_output_W[old_vocab[w]]
                model.output.b.data[idx] = old_output_b[old_vocab[w]]
                count += 1
    print('{}/{} word embs are exported'.format(count, len(old_vocab)))
    print('{}/{} word embs are imported'.format(count, len(new_vocab)))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--layer', type=int, default=2)

    parser.add_argument('--vocab', required=True)
    parser.add_argument('--new-vocab', required=True)

    parser.add_argument('--resume')
    parser.add_argument('--suffix', required=True)

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    vocab = json.load(open(args.new_vocab))
    n_vocab = len(vocab)
    print('vocab is loaded', args.new_vocab)
    print('vocab =', n_vocab)
    new_vocab = vocab

    vocab = json.load(open(args.vocab))
    n_vocab = len(vocab)
    print('old vocab is loaded', args.vocab)
    print('old vocab =', n_vocab)
    old_vocab = vocab

    model = nets.BiLanguageModel(
        n_vocab, args.unit, args.layer, 0.)

    print('load')
    chainer.serializers.load_npz(args.resume, model)

    print('alter')
    model = alter_embedding(model, old_vocab, new_vocab)

    print('save')
    chainer.serializers.save_npz(
        args.resume + '.' + args.suffix, model)


if __name__ == '__main__':
    main()
