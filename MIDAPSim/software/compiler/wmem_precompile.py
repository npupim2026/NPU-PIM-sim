from __future__ import annotations

import logging
import math
import numpy as np
from collections import OrderedDict
from functools import reduce
from software.compiler.align_compile.method import StdConvAlign
from typing import TYPE_CHECKING

from config import cfg
from logger import init_logger
from software.compiler.wmem_info import ComputeType, WMEMInfo
from software.network.types import OpType

from . import CompileTechnique

if TYPE_CHECKING:
    from software.compiler.compile_info import CompileInfo
    from software.network.model import Layer


__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"


logger = init_logger("WMEM Precompile", logging.DEBUG)


class WMEMPreCompile(
    CompileTechnique
):  # TODO: Quantization must be finished before the model is given as an input
    @classmethod
    def compile(cls, info: CompileInfo):
        model = info.model
        wmem_info_dict = info.wmem_info_dict
        func = lambda l: cls.__add_wmem_info(l, wmem_info_dict)
        model.traverse(func, topological=True)
        info.wmem_info_dict = wmem_info_dict
        return info

    @staticmethod
    def __add_wmem_info(layer: Layer, wmem_info_dict: OrderedDict[str, WMEMInfo]):
        if layer.dummy:
            return

        def _pair_compare(x, y):
            if x[1] and not y[1]:
                return x
            if y[1] and not x[1]:
                return y
            if y[0] > x[0]:
                return y
            return x

        num_wmem = cfg.MIDAP.WMEM.NUM
        op = layer.op
        filter_size = 0
        load_filter_once = False
        group_size = 1
        if op.is_conv or op.is_test:
            if op.weight is not None:
                weight = op.weight
                filter_size = weight.data[0].size
                num_filters = weight.data.shape[0]
            else:
                weight = layer.in_vtensors[-1]
                if op.type == OpType.MatMul:
                    num_filters = weight.shape[-1] // cfg.MIDAP.SYSTEM_WIDTH
                    filter_size = weight.total_size // num_filters
                else:
                    filter_size = np.prod(weight.shape[-2:])
                    num_filters = weight.shape[1]
            num_filters_per_cim = (
                num_filters // num_wmem if op.type in [OpType.StdConv, OpType.MatMulTrans, OpType.Test]
                else num_filters
            )
            wmem_size = (
                cfg.MIDAP.WMEM.NUM_ENTRIES if op.type in [OpType.StdConv, OpType.MatMulTrans, OpType.Test]
                else cfg.MIDAP.WMEM.E_NUM_ENTRIES
            )

            candidates = []
            gs = 1
            while gs <= num_filters_per_cim:
                if gs * filter_size > wmem_size:
                    break
                group_size = gs * filter_size
                num_groups = math.ceil(num_filters_per_cim / gs)
                lfo = group_size * math.ceil(num_groups / 2) <= wmem_size
                candidates.append([gs, lfo])
                if gs < num_filters_per_cim and gs * 2 > num_filters_per_cim:
                    gs = num_filters_per_cim
                else:
                    gs = gs * 2
            if not candidates:
                raise RuntimeError(f"Cannot pre-compile wmem for {layer}")
            sel = reduce(lambda cur, acc: _pair_compare(acc, cur), candidates)
            logger.debug(f"Layer {layer}, Num filters = {num_filters_per_cim}, filter size = {filter_size}, wmem_size = {wmem_size}")
            logger.debug(f"Candidates : {candidates}, Select : {sel}")
            group_size, load_filter_once = sel
        elif op.type == OpType.WeightedSum and op.weight is not None: #FIXME: Temporal solution
            load_filter_once = True
        elif op.broadcast == True:
            load_filter_once = True
        if op.is_dummy:
            return
        wmem_info = WMEMInfo(
            load_filter_once=load_filter_once,
            filter_group_size=group_size,
            compute_type=op.compute_type
        )
        wmem_info_dict[layer.name] = wmem_info
