#!/usr/bin/env python

from __future__ import print_function

import argparse
import datetime
import json
import os
import numpy

import chainer
from chainer import training
from chainer.training import extensions

import nets as bilm_nets

from text_classification import nets as class_nets
from text_classification.nlp_utils import convert_seq
from text_classification import text_datasets

from evaluator import MicroEvaluator

import args_of_text_classifier
from utils import UnkDropout


def main():
    parser = args_of_text_classifier.get_basic_arg_parser()
    args = parser.parse_args()

    print(json.dumps(args.__dict__, indent=2))
    train(args)


def train(args):
    chainer.CHAINER_SEED = args.seed
    numpy.random.seed(args.seed)

    if args.resume_vocab:
        print('load vocab from {}'.format(args.resume_vocab))
        vocab = json.load(open(args.resume_vocab))
    else:
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
    if args.validation:
        real_test = test
        dataset_pairs = chainer.datasets.get_cross_validation_datasets_random(
            train, 10, seed=777)
        train, test = dataset_pairs[0]

    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    n_class = len(set([int(d[1]) for d in train]))
    print('# class: {}'.format(n_class))

    chainer.CHAINER_SEED = args.seed
    numpy.random.seed(args.seed)
    train = UnkDropout(train, vocab['<unk>'], 0.01)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Setup a model
    chainer.CHAINER_SEED = args.seed
    numpy.random.seed(args.seed)
    if args.model == 'rnn':
        Encoder = class_nets.RNNEncoder
    elif args.model == 'cnn':
        Encoder = class_nets.CNNEncoder
    elif args.model == 'bow':
        Encoder = class_nets.BOWMLPEncoder
    encoder = Encoder(n_layers=args.layer, n_vocab=len(vocab),
                      n_units=args.unit, dropout=args.dropout)
    model = class_nets.TextClassifier(encoder, n_class)

    if args.bilm:
        bilm = bilm_nets.BiLanguageModel(
            len(vocab), args.bilm_unit, args.bilm_layer, args.bilm_dropout)
        n_labels = len(set([int(v[1]) for v in test]))
        print('# labels =', n_labels)
        if not args.no_label:
            print('add label')
            bilm.add_label_condition_nets(n_labels, args.bilm_unit)
        else:
            print('not using label')
        chainer.serializers.load_npz(args.bilm, bilm)
        with model.encoder.init_scope():
            initialW = numpy.array(model.encoder.embed.W.data)
            del model.encoder.embed
            model.encoder.embed = bilm_nets.PredictiveEmbed(
                len(vocab), args.unit, bilm, args.dropout,
                initialW=initialW)
            model.encoder.use_predict_embed = True

            model.encoder.embed.setup(
                mode=args.bilm_mode,
                temp=args.bilm_temp,
                word_lower_bound=0.,
                gold_lower_bound=0.,
                gumbel=args.bilm_gumbel,
                residual=args.bilm_residual,
                wordwise=args.bilm_wordwise,
                add_original=args.bilm_add_original,
                augment_ratio=args.bilm_ratio,
                ignore_unk=vocab['<unk>'])

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
        model.xp.random.seed(args.seed)
    chainer.CHAINER_SEED = args.seed
    numpy.random.seed(args.seed)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(args.learning_rate)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer,
        converter=convert_seq, device=args.gpu)

    from triggers import FailMaxValueTrigger
    stop_trigger = FailMaxValueTrigger(
        key='validation/main/accuracy', trigger=(1, 'epoch'),
        n_times=args.stop_epoch, max_trigger=args.epoch)
    trainer = training.Trainer(
        updater, stop_trigger, out=args.out)

    # Evaluate the model with the test dataset for each epoch
    # VALIDATION SET
    trainer.extend(MicroEvaluator(
        test_iter, model,
        converter=convert_seq, device=args.gpu))

    if args.validation:
        # REAL TEST DATASET
        real_test_iter = chainer.iterators.SerialIterator(
            real_test, args.batchsize,
            repeat=False, shuffle=False)
        eval_on_real_test = MicroEvaluator(
            real_test_iter, model,
            converter=convert_seq, device=args.gpu)
        eval_on_real_test.default_name = 'test'
        trainer.extend(eval_on_real_test)

    # Take a best snapshot
    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/accuracy', (1, 'epoch'))
    if args.save_model:
        trainer.extend(extensions.snapshot_object(
            model, 'best_model.npz'),
            trigger=record_trigger)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy',
         'test/main/loss', 'test/main/accuracy',
         'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
