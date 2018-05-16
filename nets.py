#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import argparse
import json
import warnings

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import reporter

embed_init = chainer.initializers.Uniform(.25)


def embed_seq_batch(embed, seq_batch, dropout=0., context=None):
    x_len = [len(seq) for seq in seq_batch]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(seq_batch, axis=0))
    ex = F.dropout(ex, dropout)
    if context is not None:
        ids = [embed.xp.full((l, ), i).astype('i')
               for i, l in enumerate(x_len)]
        ids = embed.xp.concatenate(ids, axis=0)
        cx = F.embed_id(ids, context)
        ex = F.concat([ex, cx], axis=1)
    exs = F.split_axis(ex, x_section, 0)
    return exs


class NormalOutputLayer(L.Linear):

    def __init__(self, *args, **kwargs):
        super(NormalOutputLayer, self).__init__(*args, **kwargs)

    def output_and_loss(self, h, t, reduce='mean'):
        logit = self(h)
        return F.softmax_cross_entropy(
            logit, t, normalize=False, reduce=reduce)

    def output(self, h, t=None):
        return self(h)


class MLP(chainer.Chain):
    def __init__(self, n_hidden, in_units, hidden_units, out_units, dropout=0.):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(in_units, hidden_units)
            self.lo = L.Linear(hidden_units, out_units)
            for i in range(2, n_hidden + 2):
                setattr(self, 'l{}'.format(i),
                        L.Linear(hidden_units, hidden_units))
        self.n_hidden = n_hidden
        self.dropout = dropout

    def __call__(self, x, label=None):
        x = self.l1(x)
        for i in range(2, self.n_hidden + 2):
            x = F.relu(x)
            x = F.dropout(x, self.dropout)
            x = getattr(self, 'l{}'.format(i))(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.lo(x)
        x = F.relu(x)
        if hasattr(self, 'l1_label') and label is not None:
            x += self.l1_label(label)
        return x


class BiLanguageModel(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_layers=2, dropout=0.5):
        super(BiLanguageModel, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            RNN = L.NStepLSTM
            self.encoder_fw = RNN(n_layers, n_units, n_units, dropout)
            self.encoder_bw = RNN(n_layers, n_units, n_units, dropout)
            self.output = NormalOutputLayer(n_units, n_vocab)
            self.mlp = MLP(1, n_units * 2, n_units, n_units, dropout)
        self.dropout = dropout
        self.n_units = n_units
        self.n_layers = n_layers

    def add_label_condition_nets(self, n_labels, label_units):
        with self.init_scope():
            self.mlp.add_link(
                'l1_label',
                L.Linear(None, self.mlp.l1.b.size, nobias=True,
                         initialW=chainer.initializers.Uniform(0.4)))
        self.n_labels = n_labels

    def encode(self, seq_batch, labels=None):
        seq_batch_wo_2bos = [seq[2::] for seq in seq_batch]
        revseq_batch_wo_2bos = [seq[::-1] for seq in seq_batch_wo_2bos]
        seq_batch_wo_2eos = [seq[:-2] for seq in seq_batch]
        bwe_seq_batch = self.embed_seq_batch(revseq_batch_wo_2bos)
        fwe_seq_batch = self.embed_seq_batch(seq_batch_wo_2eos)
        bwt_out_batch = self.encode_seq_batch(
            bwe_seq_batch, self.encoder_bw)[-1]
        fwt_out_batch = self.encode_seq_batch(
            fwe_seq_batch, self.encoder_fw)[-1]
        revbwt_concat = F.concat(
            [b[::-1] for b in bwt_out_batch], axis=0)
        fwt_concat = F.concat(fwt_out_batch, axis=0)
        t_out_concat = F.concat([fwt_concat, revbwt_concat], axis=1)
        t_out_concat = F.dropout(t_out_concat, self.dropout)
        if hasattr(self.mlp, 'l1_label') and labels is not None:
            labels = [[labels[i]] * f.shape[0]
                      for i, f in enumerate(fwt_out_batch)]
            labels = self.xp.concatenate(sum(labels, []), axis=0)
            label_concat = self.xp.zeros(
                (t_out_concat.shape[0], self.n_labels)).astype('f')
            label_concat[self.xp.arange(len(labels)), labels] = 1.
            t_out_concat = self.mlp(t_out_concat, label_concat)
        else:
            t_out_concat = self.mlp(t_out_concat)
        return t_out_concat

    def embed_seq_batch(self, x_seq_batch, context=None):
        e_seq_batch = embed_seq_batch(
            self.embed, x_seq_batch,
            dropout=self.dropout,
            context=context)
        return e_seq_batch

    def encode_seq_batch(self, e_seq_batch, encoder):
        hs, cs, y_seq_batch = encoder(None, None, e_seq_batch)
        return hs, cs, y_seq_batch

    def calculate_loss(self, input_chain, **args):
        seq_batch = sum(input_chain, [])
        t_out_concat = self.encode(seq_batch)
        seq_batch_mid = [seq[1:-1] for seq in seq_batch]
        seq_mid_concat = F.concat(seq_batch_mid, axis=0)
        n_tok = sum(len(s) for s in seq_batch_mid)
        loss = self.output_and_loss_from_concat(
            t_out_concat, seq_mid_concat,
            normalize=n_tok)
        reporter.report({'perp': self.xp.exp(loss.data)}, self)
        return loss

    def output_and_loss_from_concat(self, y, t, normalize=None):
        y = F.dropout(y, ratio=self.dropout)
        loss = self.output.output_and_loss(y, t)
        if normalize is not None:
            loss *= 1. * t.shape[0] / normalize
        else:
            loss *= t.shape[0]
        return loss

    def calculate_loss_with_labels(self, seq_batch_with_labels):
        seq_batch, labels = seq_batch_with_labels
        t_out_concat = self.encode(seq_batch, labels=labels)
        seq_batch_mid = [seq[1:-1] for seq in seq_batch]
        seq_mid_concat = F.concat(seq_batch_mid, axis=0)
        n_tok = sum(len(s) for s in seq_batch_mid)
        loss = self.output_and_loss_from_concat(
            t_out_concat, seq_mid_concat,
            normalize=n_tok)
        reporter.report({'perp': self.xp.exp(loss.data)}, self)
        return loss

    def predict(self, xs, labels=None):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            t_out_concat = self.encode(xs, labels=labels, add_original=0.)
            prob_concat = F.softmax(self.output.output(t_out_concat)).data
            x_len = [len(x) for x in xs]
            x_section = np.cumsum(x_len[:-1])
            ps = np.split(cuda.to_cpu(prob_concat), x_section, 0)
        return ps

    def predict_embed(self,
                      xs, embedW,
                      labels=None,
                      dropout=0.,
                      mode='sampling',
                      temp=1.,
                      word_lower_bound=0.,
                      gold_lower_bound=0.,
                      gumbel=True,
                      residual=0.,
                      wordwise=True,
                      add_original=0.,
                      augment_ratio=0.25):
        x_len = [len(x) for x in xs]
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            t_out_concat = self.encode(xs, labels=labels)
            prob_concat = self.output.output(t_out_concat).data
            prob_concat /= temp
            prob_concat += self.xp.random.gumbel(
                size=prob_concat.shape).astype('f')
            prob_concat = F.softmax(prob_concat).data

        out_concat = F.embed_id(
            self.xp.argmax(prob_concat, axis=1).astype(np.int32), embedW)

        # insert eos
        eos = embedW[0][None]
        new_out = []
        count = 0
        for i, x in enumerate(xs):
            new_out.append(eos)
            new_out.append(out_concat[count:count + len(xs) - 2])
            new_out.append(eos)
            count += len(xs) - 2
        out_concat = F.concat(new_out, axis=0)

        def embed_func(x): return F.embed_id(x, embedW, ignore_label=-1)
        raw_concat = F.concat(
            sequence_embed(embed_func, xs, self.dropout), axis=0)
        b, u = raw_concat.shape

        mask = self.xp.broadcast_to(
            (self.xp.random.rand(b, 1) < augment_ratio),
            raw_concat.shape)
        out_concat = F.where(mask, out_concat, raw_concat)

        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        out_concat = F.dropout(out_concat, dropout)
        exs = F.split_axis(out_concat, x_section, 0)
        return exs


def sequence_embed(embed, xs, dropout=0.):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


def block_embed(embed, x, dropout=0.):
    """Embedding function followed by convolution

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which
            is a :math:`(B, L)`-shaped int array. Its first dimension
            :math:`(B)` is assumed to be the *minibatch dimension*.
            The second dimension :math:`(L)` is the length of padded
            sentences.
        dropout (float): Dropout ratio.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(B, N, L, 1)`. :math:`(N)` is the number of dimensions
        of word embedding.

    """
    e = embed(x)
    e = F.dropout(e, ratio=dropout)
    e = F.transpose(e, (0, 2, 1))
    e = e[:, :, :, None]
    return e


class PredictiveEmbed(chainer.Chain):
    def __init__(self, n_vocab, n_units, bilm,
                 dropout=0., initialW=embed_init):
        super(PredictiveEmbed, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, ignore_label=-1,
                                   initialW=initialW)
            self.bilm = bilm
        self.n_vocab = n_vocab
        self.n_units = n_units
        self.dropout = dropout

    def __call__(self, x):
        return self.embed(x)

    def setup(self,
              mode='weighted_sum',
              temp=1.,
              word_lower_bound=0.,
              gold_lower_bound=0.,
              gumbel=True,
              residual=0.,
              wordwise=True,
              add_original=1.,
              augment_ratio=0.5,
              ignore_unk=-1):
        self.config = {
            'dropout': self.dropout,
            'mode': mode,
            'temp': temp,
            'word_lower_bound': 0.,
            'gold_lower_bound': 0.,
            'gumbel': gumbel,
            'residual': residual,
            'wordwise': wordwise,
            'add_original': add_original,
            'augment_ratio': augment_ratio
        }
        if ignore_unk >= 0:
            self.bilm.output.b.data[ignore_unk] = -1e5

    def embed_xs(self, xs, batch='concat'):
        if batch == 'concat':
            x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
            ex_block = block_embed(self.embed, x_block, self.dropout)
            return ex_block
        elif batch == 'list':
            exs = sequence_embed(self.embed, xs, self.dropout)
            return exs
        else:
            raise NotImplementedError

    def embed_xs_with_prediction(self, xs, labels=None, batch='concat'):
        predicted_exs = self.bilm.predict_embed(
            xs, self.embed.W,
            labels=labels,
            dropout=self.config['dropout'],
            mode=self.config['mode'],
            temp=self.config['temp'],
            word_lower_bound=self.config['word_lower_bound'],
            gold_lower_bound=self.config['gold_lower_bound'],
            gumbel=self.config['gumbel'],
            residual=self.config['residual'],
            wordwise=self.config['wordwise'],
            add_original=self.config['add_original'],
            augment_ratio=self.config['augment_ratio'])
        if batch == 'concat':
            predicted_ex_block = F.pad_sequence(predicted_exs, padding=0.)
            predicted_ex_block = F.transpose(
                predicted_ex_block, (0, 2, 1))[:, :, :, None]
            return predicted_ex_block
        elif batch == 'list':
            return predicted_exs
        else:
            raise NotImplementedError
