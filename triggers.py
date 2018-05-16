import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer import reporter
from chainer.training import util
from chainer import utils
from chainer.utils import type_check
from chainer.training import triggers


class FailBestValueTrigger(triggers.IntervalTrigger):

    """Trigger invoked when specific value fails to become best.

    Args:
        key (str): Key of value.
        compare (function): Compare function which takes current best value and
            new value and returns whether new value is better than current
            best.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.

    """

    def __init__(self, key, compare, trigger=(1, 'epoch'),
                 n_times=5, max_trigger=None, print_triger=False):
        self.period = max_trigger
        self.unit = 'epoch'

        self._key = key
        self._best_value = None
        self._interval_trigger = util.get_trigger(trigger)
        self._init_summary()
        self._compare = compare
        self._print_triger = print_triger

        self._n_times = n_times
        self._n_fails = 0

        self._max_trigger = max_trigger
        self._n_triggers = 0

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.

        Args:
            trainer (~chainer.training.Trainer): Trainer object that this
                trigger is associated with. The ``observation`` of this trainer
                is used to determine if the trigger should fire.

        Returns:
            bool: ``True`` if the corresponding extension should be invoked in
                this iteration.

        """

        observation = trainer.observation
        summary = self._summary
        key = self._key
        if key in observation:
            summary.add({key: observation[key]})

        if not self._interval_trigger(trainer):
            return False

        stats = summary.compute_mean()
        value = float(stats[key])  # copy to CPU
        self._init_summary()

        self._n_triggers += 1
        if self._n_triggers == 1:
            return False
        if self._max_trigger is not None \
           and self._n_triggers >= self._max_trigger:
            return True

        if self._best_value is None or self._compare(self._best_value, value):
            self._best_value = value
            self._n_fails = 0
            return False
        self._n_fails += 1
        if self._n_fails >= self._n_times:
            return True
        else:
            return False

    def _init_summary(self):
        self._summary = reporter.DictSummary()


class FailMaxValueTrigger(FailBestValueTrigger):
    def __init__(self, key, trigger=(1, 'epoch'), n_times=5, max_trigger=None):
        super(FailMaxValueTrigger, self).__init__(
            key, lambda max_value, new_value: new_value > max_value,
            trigger, n_times, max_trigger)


class FailMinValueTrigger(FailBestValueTrigger):
    def __init__(self, key, trigger=(1, 'epoch'), n_times=5, max_trigger=None):
        super(FailMinValueTrigger, self).__init__(
            key, lambda min_value, new_value: new_value < min_value,
            trigger, n_times, max_trigger)
