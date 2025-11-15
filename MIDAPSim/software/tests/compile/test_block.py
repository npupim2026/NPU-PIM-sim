from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from models.model_builder import ModelBuilder
from software.network import networks
from software.compiler.align_compile import AlignCompile
from software.compiler.block_compile import BlockCompile
from software.compiler.compile_info import CompileInfo
from software.compiler.compiler import CompileTechnique
from software.generic_op.generic_model import GenericModel
from software.network.model import ModelGraph

if TYPE_CHECKING:
    from typing import List


class TestBlock(object):
    @staticmethod
    def mb2model(mb: ModelBuilder, test_name: str):
        odict = mb.get_operator_dict()
        cv = GenericModel()
        cv.operator_dict = odict
        cv.post_process()

        model = ModelGraph(test_name)
        model.build(odict)
        return model

    @staticmethod
    def compile_and_align(model) -> CompileInfo:
        info = CompileInfo(model)
        compilers: List[CompileTechnique] = [AlignCompile, BlockCompile]
        for compiler in compilers:
            info = compiler.compile(info)
        return info

    # @pytest.mark.parametrize("net_name", networks.keys())
    @pytest.mark.parametrize("net_name", ["mobilenet", "mobilenet_v2", "resnet50"])
    def test_block_builder(self, net_name: str):
        mb = networks[net_name](None)
        model = self.mb2model(mb, net_name)
        info = self.compile_and_align(model)
        assert len(model) == len(info.layer_dict)
        visit_checker = dict()
        for l in info.layer_dict.keys():
            if l not in visit_checker:
                assert all(i in visit_checker for i in l.inputs)
                visit_checker[l] = True
