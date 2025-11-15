from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from sys import maxsize
from typing import TYPE_CHECKING

import attr
from logger import init_logger

if TYPE_CHECKING:
    from typing import Any, Callable, Generator

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"

logger = init_logger("Optimizer", logging.INFO)


@attr.s(slots=True)
class OptItem(object):
    item: Any = attr.ib()
    cost: int = attr.ib()


class Optimizer(ABC):
    __slots__ = ("_calc_func", "_space", "_best_item")

    def __init__(self, calc_func: Callable[[Any], Any], space: Generator):
        self._calc_func = calc_func
        self._space = space
        self._best_item: OptItem = None

    def optimize(self):
        for s in self._space:
            item, value = self._calc_func(s)
            self._compare_and_set(item, value)

    @abstractmethod
    def _compare_and_set(self, item, value):
        raise NotImplementedError

    @property
    def best(self) -> OptItem:
        return self._best_item

    @property
    def item(self):
        return self._best_item.item

    @property
    def cost(self):
        return self._best_item.cost


class Minimizer(Optimizer):
    def __init__(self, calc_func: Callable[[Any], Any], space: Generator):
        super(Minimizer, self).__init__(calc_func, space)
        self._best_item = OptItem(None, maxsize)

    def _compare_and_set(self, solution, value: int):
        if self.cost > value:  # minimize
            self._best_item = OptItem(solution, value)
