from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from ordered_set import OrderedSet

if TYPE_CHECKING:
    from typing import Any, Callable, Generator


__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang", "Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class Graph(object):
    def __init__(self):
        self._graph = defaultdict(set)
        self._items: OrderedSet[Any] = OrderedSet()
        self._num_in_edges = defaultdict(lambda: 0)

    @property
    def num_node(self):
        return len(self._graph)

    def add(self, src, dest):
        self._graph[src].add(dest)
        self._items.update([src, dest])
        self._num_in_edges[dest] += 1

    def add_node(self, node):
        self._items.add(node)

    def __iter__(self) -> Generator[Any, None, None]:
        for item in self._items:
            yield item

    def remove(self, item):
        for _, cxns in self._graph.items():
            try:
                cxns.remove(item)
            except KeyError:
                pass
        try:
            for dst in self._graph[item]:
                self._num_in_edges[dst] -= 1
            del self._graph[item]
        except KeyError:
            pass
        self._items.discard(item)

    def traverse(self, func: Callable[[Any], Any], topological: bool = False):
        if topological:
            self._topological_traverse(func)
        else:
            self._traverse(func)

    def _traverse(self, func: Callable[[Any], Any]):
        for item in self._items:
            func(item)

    def _topological_traverse(self, func: Callable[[Any], Any]):
        from copy import copy

        queue = []
        for n in self:
            if self._num_in_edges[n] == 0:
                queue.append(n)

        visit_count = copy(self._num_in_edges)
        while queue:
            cur = queue.pop()
            func(cur)

            for n in self._graph[cur]:
                visit_count[n] -= 1
                if visit_count[n] == 0:
                    queue.append(n)

    def get_next_node(self, src):
        return self._graph[src]

    def get_incoming_nodes(self, node):
        incoming_nodes = []
        for src in self._items:
            if node in self.get_next_node(src):
                incoming_nodes.append(src)
        return incoming_nodes
