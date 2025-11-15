from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from logger import init_logger
from software.compiler import CompileTechnique
from software.compiler.layer_compile.fmem_info import FMEMInfo
from software.compiler.optimizer import Minimizer

from .intermediate_info import IntermediateInfo
from .min_policy import MinPolicy
from .mapping import Mapping


if TYPE_CHECKING:
    from typing import Generator, Type

    from software.compiler.compile_info import CompileInfo

    from .layer_info import LayerInfo
    from .policy import LayerCompliePolicy

__author__ = "Duseok Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Duseok Kang"]

__license__ = "MIT License"
__maintainer__ = "Duseok Kang"
__status__ = "Production"

logger = init_logger("Layer Compile", logging.DEBUG)


class LayerCompile(CompileTechnique):
    policy: Type[LayerCompliePolicy] = MinPolicy

    @staticmethod
    def _print_status(curr: LayerInfo, fmem: FMEMInfo, func=logger.debug):
        def _print_func(e):
            return [str(x) for x in e]

        rw_str = " (R)" if curr.reversed else " (O)"
        func(f"=================== {curr.name + rw_str:^12s} ===================")
        func(f"P   : {[t.name for t in curr.layer.in_vtensors]}")
        func(f"B   : {_print_func(curr.behavior)}")
        func(f"IN  : {_print_func(curr.mapping.input)} {curr.in_tensor.flip_x}")
        func(f"OUT : {_print_func(curr.mapping.output)} {curr.out_tensor.flip_x}")
        func(f"S   : {curr.stationary}")
        func(f"FMEM: {fmem}")
        func("====================================================")

    @classmethod
    def set_policy(cls, policy: Type[LayerCompliePolicy]):
        cls.policy = policy

    @classmethod
    def compile(cls, info: CompileInfo) -> bool:
        for curr in info.layers[info.num_compiled :]:
            input = info.get_layer_info(curr.layer.input) if curr.layer.input else None
            if not cls._compile_layer(info, input, curr):
                return False
        info.num_compiled = len(info.layers)
        return True

    @staticmethod
    def outbank_space(num_available_fmem: int, out_stationary: int) -> Generator[int]:
        if out_stationary == -1:
            for num_outbanks in reversed(range(num_available_fmem)):
                yield num_outbanks
        else:
            yield out_stationary

    @classmethod
    def _compile_layer(cls, info: CompileInfo, input: LayerInfo, curr: LayerInfo):
        if curr.dummy:
            mapping = curr.mapping
            mapping.input = mapping.output = input.mapping.output if input else ()
            return True
        if curr.reduction: ## FIXME: Temporal code for reduction layer
            mapping = curr.mapping
            d = curr.op_info.in2out.input2output(curr, [], -1)[0]
            bank = info.fmem_info.write(d)
            mapping.input = input.mapping.output if input else ()
            if bank is not None:
                mapping.output.append(Mapping(bank, d))
            return True
        
        # print(f"Compile {curr.name}")
        fmem = info.fmem_info
        _info = IntermediateInfo.from_fmem(fmem, curr.ordered_input_vtensors)
        space = cls.outbank_space(fmem.num_available_banks + len(_info.mapping.input), curr.stationary.output)
        curr.mapping.input = _info.mapping.input

        def search(num_outbanks):
            num_banks = {"input": curr.stationary.input, "output": num_outbanks}
            optimizer = cls.policy(curr, fmem, num_banks)
            optimizer.search(_info)
            return optimizer.best, optimizer.best.total_size

        optimizer = Minimizer(search, space)
        optimizer.optimize()
        best = optimizer.item

        if best:
            info.fmem_info = cls.reverse_data(best.fmem, curr)
            curr.behavior = best.action
            curr.mapping = best.mapping
            cls._print_status(curr, info.fmem_info, print)
            return True
        return False

    @staticmethod
    def reverse_data(fmem: FMEMInfo, info: LayerInfo):
        if not info.reversed:
            return fmem
        tensor = info.out_tensor
        width = tensor.width
        for d in fmem:
            if d.tensor == tensor:
                d.pivot = width - d.last_x
        return fmem
