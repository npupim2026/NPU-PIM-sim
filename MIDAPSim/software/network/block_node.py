from __future__ import annotations

from typing import TYPE_CHECKING

from . import Layer

if TYPE_CHECKING:
    from typing import List


__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


class BlockNode(object):
    __slots__ = ["_name", "_source", "_main_path", "_sub_paths"]

    def __init__(
        self, source: Layer, main_path: List[Layer], sub_paths: List[List[Layer]]
    ):

        self._name = f"Block({source.name})"
        self._source = source
        self._main_path = main_path
        self._sub_paths = sub_paths

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> Layer:
        return self._source

    @property
    def sink(self) -> Layer:
        if self._main_path:
            return self._main_path[-1]
        return self._source

    @property
    def main_path(self) -> List[Layer]:
        return self._main_path

    @property
    def sub_paths(self) -> List[List[Layer]]:
        return self._sub_paths

    @property
    def paths(self) -> List[List[Layer]]:
        paths = [path for path in self._sub_paths if path]
        if not self.is_leaf or len(paths) == 0:
            paths = [self._main_path] + paths
        return paths

    @property
    def all_paths(self) -> List[List[Layer]]:
        paths = self.paths
        return [[self.source]] + paths

    @property
    def is_leaf(self) -> bool:
        return len(self._main_path) == 0 and len(self._source.outputs) == len(
            self._sub_paths
        )

    @property
    def layers(self) -> List[Layer]:
        layers = [self._source] + self._main_path
        for sub_path in self._sub_paths:
            layers += sub_path
        return layers

    @property
    def macs(self) -> int:
        layers: List[Layer] = self.layers
        mac_count = 0
        for layer in layers:
            mac_count += layer.op.macs
        return mac_count

    def __repr__(self) -> str:
        return f"[{self.name}, ({self.paths})]"
