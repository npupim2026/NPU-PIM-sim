from __future__ import annotations

from sys import maxsize
from typing import TYPE_CHECKING

from graphviz import Digraph

from . import Layer, LayerBlock

if TYPE_CHECKING:
    from typing import Any, Callable, List

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class LayerPath(object):
    def __init__(self, path: List[Layer | LayerBlock]):
        self._path = path
        self._stationary = [maxsize, maxsize]
        self._layers = [v for v in path if isinstance(v, Layer)]
        self._blocks = [v for v in path if isinstance(v, LayerBlock)]

    def get_last_layer(self):
        v = self._path[-1]
        return v if isinstance(v, Layer) else v.sink

    def draw(self, g: Digraph):
        draw_layer: Callable[[Layer], None] = lambda l: self.__draw_layer(g, l)
        draw_block: Callable[[LayerBlock], None] = lambda b: self.__draw_block(g, b)
        self.traverse(layer_func=draw_layer, block_func=draw_block)

    def __len__(self):
        return len(self._path)

    @property
    def input_stationary(self) -> int:
        return self._stationary[0]

    @input_stationary.setter
    def input_stationary(self, statinoary: int) -> int:
        self._stationary[0] = statinoary

    @property
    def output_stationary(self) -> int:
        return self._stationary[1]

    @output_stationary.setter
    def output_stationary(self, statinoary: int) -> int:
        self._stationary[1] = statinoary

    @staticmethod
    def __draw_layer(g: Digraph, layer: Layer):
        for next in layer.parent.get_next_node(layer):
            if next in layer.outputs:
                g.edge(layer.name, next.name, label=layer.out_vtensor.name)
            else:
                g.edge(layer.name, next.name, label=layer.out_vtensor.name, style="dashed")

    @staticmethod
    def __draw_block(g: Digraph, block: LayerBlock):
        # draw block with subgraph
        with g.subgraph(name=f"cluster_{block.name}") as sg:
            block.draw_except_source(sg)
            sg.attr(label=block.name)

    def __traverse_ordered(
        self,
        layer_func: Callable[[Layer], Any],
        block_func: Callable[[LayerBlock], Any],
    ):
        for v in self:
            if isinstance(v, Layer):
                layer_func(v)
            elif isinstance(v, LayerBlock):
                block_func(v)

    def __traverse(
        self,
        layer_func: Callable[[Layer], Any],
        block_func: Callable[[LayerBlock], Any],
    ):
        l: Layer
        for l in self._layers:
            layer_func(l)
        b: LayerBlock
        for b in self._blocks:
            block_func(b)

    def traverse(self, ordered: bool = True, **kwargs):
        if ordered:
            self.__traverse_ordered(**kwargs)
        else:
            self.__traverse(**kwargs)

    def __iter__(self):
        return iter(self._path)

    def __getitem__(self, item):
        return self._path[item]

    def __repr__(self) -> str:
        return f"{[l.name for l in self._path]}"
