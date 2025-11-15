from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from models.model_builder import ModelBuilder
from software.compiler.align_compile import AlignCompile
from software.compiler.compile_info import CompileInfo
from software.generic_op.generic_model import GenericModel
from software.network.model import ModelGraph

if TYPE_CHECKING:
    from typing import Dict, Tuple


class TestAlign(object):
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
    def compile_and_align(model) -> None:
        compile_info = CompileInfo(model)
        AlignCompile.compile(compile_info)

    @pytest.mark.parametrize(
        "shape,conv_args,expected",
        [
            (
                (1, 3, 224, 224),
                {
                    "out_c": 32,
                    "k": 3,
                    "stride": 1,
                    "pad": "same",
                    "dilation": 1,
                },
                {
                    "input": (50176, 1, 64),
                    "weight": (32, 1, 1, 64),
                    "output": (224, 224, 32),
                },
            ),
            (
                (1, 3, 224, 224),
                {
                    "out_c": 30,
                    "k": 3,
                    "stride": 1,
                    "pad": "same",
                    "dilation": 1,
                },
                {
                    "input": (50176, 1, 64),
                    "weight": (32, 1, 1, 64),
                    "output": (224, 224, 32),
                },
            ),
            (
                (1, 3, 224, 224),
                {
                    "out_c": 32,
                    "k": 5,
                    "stride": 2,
                    "pad": "same",
                    "dilation": 1,
                },
                {
                    "input": (12544, 1, 128),
                    "weight": (32, 1, 1, 128),
                    "output": (112, 112, 32),
                },
            ),
        ],
    )
    def test_gemm_alignment(self, shape: Tuple, conv_args: Dict, expected: Tuple):
        test_name = "test_gemm_alignment"
        mb = ModelBuilder(test_name)
        x = mb.set_input_tensor(tensor_shape=shape)
        x = mb.Conv(x, shape[1], **conv_args, name="GEMM")
        _ = mb.Conv(x, conv_args["out_c"], conv_args["out_c"], 1, 1, "same")

        model = self.mb2model(mb, test_name)
        self.compile_and_align(model)
        gemm_layer = model._str2layer["GEMM"]
        assert gemm_layer.in_vtensors[0].shape == expected["input"]
        assert gemm_layer.out_vtensor.shape == expected["output"]
        assert gemm_layer.op.weight.shape == expected["weight"]

    @pytest.mark.parametrize(
        "shape,conv_args,expected",
        [
            (
                (1, 64, 112, 112),
                {
                    "out_c": 64,
                    "k": 3,
                    "stride": 1,
                    "pad": "same",
                    "dilation": 1,
                },
                {
                    "input": (112, 112, 64),
                    "weight": (64, 3, 192),
                    "output": (112, 112, 64),
                },
            ),
            (
                (1, 48, 112, 112),
                {
                    "out_c": 64,
                    "k": 5,
                    "stride": 2,
                    "pad": "same",
                    "dilation": 1,
                },
                {
                    "input": (112, 112, 48),
                    "weight": (64, 5, 256),
                    "output": (56, 56, 64),
                },
            ),
            (
                (1, 30, 112, 112),
                {
                    "out_c": 64,
                    "k": 3,
                    "stride": 1,
                    "pad": "same",
                    "dilation": 1,
                },
                {
                    "input": (112, 112, 32),
                    "weight": (64, 3, 128),
                    "output": (112, 112, 64),
                },
            ),
        ],
    )
    def test_stdconv_alignment(self, shape: Tuple, conv_args: Dict, expected: Tuple):
        test_name = "test_stdconv_alignment"
        mb = ModelBuilder(test_name)
        x = mb.set_input_tensor(tensor_shape=shape)
        x = mb.Conv(x, shape[1], shape[1], 3, 1, "same")
        x = mb.Conv(x, shape[1], **conv_args, name="stdconv")
        _ = mb.Conv(x, conv_args["out_c"], conv_args["out_c"], 1, 1, "same")

        model = self.mb2model(mb, test_name)
        self.compile_and_align(model)
        layer = model._str2layer["stdconv"]
        assert layer.in_vtensors[0].shape == expected["input"]
        assert layer.out_vtensor.shape == expected["output"]
        assert layer.op.weight.shape == expected["weight"]

    @pytest.mark.parametrize(
        "shape,conv_args,expected",
        [
            (
                (1, 32, 112, 112),
                {
                    "k": 3,
                    "stride": 1,
                    "pad": "same",
                    "dilation": 1,
                },
                {
                    "input": (112, 112, 64),
                    "weight": (1, 3, 192),
                    "output": (112, 112, 64),
                },
            ),
        ],
    )
    def test_dwconv_alignment(self, shape: Tuple, conv_args: Dict, expected: Tuple):
        test_name = "test_dwconv_alignment"
        mb = ModelBuilder(test_name)
        x = mb.set_input_tensor(tensor_shape=shape)
        in_c = shape[1]
        x = mb.Conv(x, in_c, in_c, 3, 1, "same")
        x = mb.DWConv(x, in_c, **conv_args, name="dwconv")
        _ = mb.Conv(x, in_c, in_c, 1, 1, "same")

        model = self.mb2model(mb, test_name)
        self.compile_and_align(model)
        layer = model._str2layer["dwconv"]
        assert layer.in_vtensors[0].shape == expected["input"]
        assert layer.out_vtensor.shape == expected["output"]
        assert layer.op.weight.shape == expected["weight"]

    # @pytest.mark.parametrize("", [])
    # def test_arith_alignment():
    #     # alignment 확인..
    #     pass
