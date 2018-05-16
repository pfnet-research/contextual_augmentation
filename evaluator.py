import collections
import copy

import six

import chainer
from chainer import configuration
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extension


class MicroEvaluator(chainer.training.extensions.Evaluator):

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        # summary = reporter_module.DictSummary()
        summary = collections.defaultdict(list)

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)
            n_data = len(batch)
            summary['n'].append(n_data)
            # summary.add(observation)
            for k, v in observation.items():
                summary[k].append(v)

        mean = dict()
        ns = summary['n']
        del summary['n']
        for k, vs in summary.items():
            mean[k] = sum(v * n for v, n in zip(vs, ns)) / sum(ns)
        return mean
        # return summary.compute_mean()
