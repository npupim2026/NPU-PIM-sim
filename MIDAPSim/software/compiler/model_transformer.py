from __future__ import annotations

import logging
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from software.network.types import OpType
from typing import TYPE_CHECKING

from logger import init_logger

from config import cfg

if TYPE_CHECKING:
    from typing import List

    from software.network.layer import Layer
    from software.network.model import ModelGraph


__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang", "Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"

logger = init_logger("Graph Compile", logging.INFO)


class ModelTransformer(ABC):
    @abstractmethod
    def transform(self, model: ModelGraph) -> ModelGraph:
        pass


class PruningHeuristic(ModelTransformer):
    def __init__(self):
        self._layer2level = {}

    def _set_level(self, model):
        nexts = model.inputs
        level = 0
        visits = defaultdict(int, {layer: layer.num_inputs - 1 for layer in model})
        while nexts:
            curs = nexts
            nexts = []
            for layer in curs:
                if visits[layer] > 0:
                    visits[layer] -= 1
                    continue
                self._layer2level[layer] = level
                nexts.extend(layer.outputs)
            level += 1

    def transform(self, model: ModelGraph) -> ModelGraph:
        self._set_level(model)
        return self.pruning_model(model)

    @staticmethod
    def _print_status(layers: List[Layer]):
        def _print_func(layer: Layer):
            return f"{layer.inputs} -> {layer} -> {layer.outputs}"

        logger.debug("====================================================")
        for layer in layers:
            logger.debug(f"{_print_func(layer)}")
        logger.debug("====================================================")

    def pruning_model(self, model: ModelGraph):
        logger.debug(f"Before pruning:")
        self._print_status(model.layers)
        layer: Layer
        for layer in model:
            # if layer.op.type == OpType.GlobalPool and not cfg.MODEL.REDUCTION_LOGIC: # FIXME: Temporal code
            #     src = layer.inputs[0]
            #     if len(src.outputs) > 1:
            #         model.remove_connection(src, layer)
            if len(layer.inputs) > 1 or layer.op.type in [OpType.HostProcess, OpType.Concat] or (len(layer.inputs) == 0 and layer.op.type == OpType.Dummy) or (len(layer.inputs) > 0 and layer.op.type in [OpType.MatMul, OpType.MatMulTrans, OpType.RoPE]):
                logger.debug(f"Pruning edge for layer {layer}")
                self.__pruning_edge(layer, model)
                if len(layer.inputs) > 1:
                    raise RuntimeError(f"Pruning error for : {layer}")
        logger.debug(f"After pruning")
        self._print_status(model.layers)
        return model

    def __pruning_edge(self, node: Layer, model: ModelGraph):
        # Determine which edges to disconnect
        from software.network.types import OpType

        if node.op.type == OpType.HostProcess:
            logger.debug("HostProcess: Disconnect all incoming edges")
            src_layers = list(node.inputs)
            dst_layers = list(node.outputs)
            for src in src_layers:
                model.remove_connection(src, node)
            for dst in dst_layers:
                model.remove_connection(node, dst)
        elif node.op.type == OpType.Concat:  # Disconnect all edges
            logger.debug("Concat: Disconnect all incoming edges")
            src_layers = list(node.inputs)
            for src in src_layers:
                model.remove_connection(src, node)
        elif node.op.type == OpType.Dummy and not node.input:
            logger.debug("First dummy layer: Disconnect all outgoing edges")
            for dst in list(node.outputs):
                model.remove_connection(node, dst)
        elif node.op.type in [OpType.MatMul, OpType.MatMulTrans, OpType.RoPE]:
            # commutative property does not hold (Temporal solution)
            if node.inputs[-1].out_tensor == node.in_vtensors[-1].tensor:
                model.remove_connection(node.inputs[-1], node)
        elif node.op.type in [OpType.Sum, OpType.WeightedSum, OpType.Mul]:
            inputs = node.inputs
            if len(inputs) != 2:
                raise ValueError(f"Weird inputs for {node}")
            if node.op.broadcast:
                # 1. disconnect broadcasting edge
                src = None
                for layer in inputs:
                    if layer.out_vtensor.width == 1:
                        src = layer
                if src is None:
                    raise ValueError(
                        f"Weird inputs for {node}: no input is given for broadcasting"
                    )
            else:
                # 2. disconnect shorter edge
                src = (
                    inputs[0]
                    if self._layer2level[inputs[0]] <= self._layer2level[inputs[1]]
                    else inputs[1]
                )
                if node.op.type == OpType.WeightedSum and node.op.weight is not None and src.output_scale / node.op.weight.data[0, 0, 1, 0] != node.input_scale:
                    node.op.weight.data = np.flip(node.op.weight.data, axis=-2)
            model.remove_connection(src, node)
            logger.debug(f"Arithmetic: pruning incoming edge {src} -> {node}")
        else:
            raise NotImplementedError
