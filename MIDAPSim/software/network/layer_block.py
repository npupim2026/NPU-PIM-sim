from __future__ import annotations

from typing import TYPE_CHECKING

from graphviz import Digraph

if TYPE_CHECKING:
    from typing import Any, Callable, List

    from . import Layer, LayerPath


__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"


class LayerBlock(object):
    __slots__ = ["_name", "_source", "_sink", "_paths"]

    def __init__(self, source: Layer, sink: Layer, paths: List[Layer | LayerBlock]):
        from . import LayerPath

        self._name = f"Block({source.name}->{sink.name})"
        self._source = source
        self._sink = sink
        self._paths = [LayerPath(p) for p in paths]

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_outputs(self) -> int:
        return self._sink.num_outputs

    @property
    def outputs(self) -> List[Layer]:
        return self._sink.outputs

    @property
    def source(self) -> Layer:
        return self._source

    @property
    def sink(self) -> Layer:
        return self._sink

    @property
    def paths(self) -> List[LayerPath]:
        return self._paths

    def draw_source_edges(self, g: Digraph):
        for l in self._source.parent.get_next_node(self._source):
            if l in self._source.outputs:
                g.edge(self._source.name, l.name, label=self._source.out_vtensor.name)
            else:
                g.edge(self._source.name, l.name, label=self._source.out_vtensor.name, style="dashed")

    def draw(self, g: Digraph = None):
        g = g if g else Digraph(self.name)
        g.attr(label=self.name, shape="box")

        self.draw_source_edges(g)
        self.draw_except_source(g)

    def draw_except_source(self, g: Digraph):
        draw_paths: Callable[[LayerPath], None] = lambda p: p.draw(g)
        self.traverse_paths(draw_paths)
        g.render(
            filename=f"{self.dot_prefix}{self.name}.gv", directory=self.dot_directory, format="png"
        )

    def traverse_paths(self, func: Callable[[LayerPath], Any]):
        for p in self._paths:
            func(p)

    def __repr__(self) -> str:
        return self.name
