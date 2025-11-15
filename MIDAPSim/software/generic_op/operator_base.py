from __future__ import annotations
from software.compiler.input2output import InData2OutData
from software.network.types import TensorType, VTensorType
from software.network.tensor import Tensor
from software.network.quant_info import QuantType, TensorQuantInfo

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from software.compiler.align_compile import Alignment
    from software.network.op_info import OpType
    from typing import Tuple


class OperatorBase(object):
    # All tensor-like features should be np.array type
    def __init__(
        self,
        name=None,
        op_type=None,
        order="NCHW",
        input_layers=[],
        output_tensor=None,
        activation=None,
        weight=None,
        bias=None,
        weight_scale=None,
        bias_scale=None,
        output_scale=None,
        act_scale_32b_to_16b=None,
        act_scale_16b_to_8b=None,
        **kwargs,
    ):
        if name is None:
            raise ValueError("name: operator name must be defined")
        if op_type is None:
            raise ValueError("op_type: operator type must be defined")
        self.name = name  # Operator name
        self.type = op_type  # Operator type
        # Shape - Default : NCHW, tensor_to_midap_tensor converts order to 'NWHC'
        self.order = order if isinstance(order, str) else "NCHW"
        self.input_layers = input_layers  # Operator input
        self.output_tensor = output_tensor  # Operator output tensor - for verification
        self.activation = activation
        self.weight = weight
        self.bias = bias
        self.next = []
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.output_scale = output_scale
        self.act_scale_32b_to_16b = act_scale_32b_to_16b
        self.act_scale_16b_to_8b = act_scale_16b_to_8b
        if len(kwargs) > 0:
            print("Warning: operator {}: parameter {} is not used".format(name, kwargs))

    def __del__(self):
        del self.input_layers, self.output_tensor, self.activation

    def get_op_type(self) -> OpType:
        raise NotImplementedError(f"Not implement get_op_type function in {self.name}")

    def get_alignment(self) -> Alignment:
        raise NotImplementedError(
            f"Not implement get_alignment function in {self.name}"
        )

    def get_in2out(self) -> InData2OutData:
        raise NotImplementedError(f"Not implement get_in2out function in {self.name}")

    @property
    def is_first_tensor(self) -> bool:
        return len(self.input_layers) == 0

    @property
    def vtype(self) -> VTensorType:
        return VTensorType.Default

    @property
    def vscale(self) -> Tuple(int, int, int):
        return (1, 1, 1)

    @property
    def voffset(self) -> Tuple(int, int, int):
        return (0, 0, 0)

    @property
    def kernel(self):
        return (1, 1)

    @property
    def pad(self):
        return (0, 0, 0, 0)

    @property
    def stride(self):
        return 1

    @property
    def dilation(self):
        return 1

    @property
    def broadcast(self):
        return False

    @property
    def in_plane(sef):
        return False

    @property
    def test_code(self):
        return None

    def get_weight_tensor(self):
        if self.weight is None:
            return None
        tensor_name = self.name + "_w"
        tensor_type = TensorType.Constant
        quant_info = None
        if self.weight_scale is not None:
            quant_info = TensorQuantInfo(type=QuantType.Symm, bits=8, scale=self.weight_scale)
        return Tensor(tensor_name, self.weight, tensor_type, quant_info)

    def get_bias_tensor(self):
        if self.bias is None:
            return None
        tensor_name = self.name + "_b"
        tensor_type = TensorType.Constant
        quant_info = None
        if self.bias_scale is not None:
            quant_info = TensorQuantInfo(type=QuantType.Symm, bits=8, scale=self.bias_scale)
        return Tensor(tensor_name, self.bias, tensor_type, quant_info)

    def tensor_to_midap_tensor(self):
        # NCHW --> NHWC
        if self.order == "NCHW":
            if len(self.output_tensor.shape) == 4:
                #     self.output_tensor = self.output_tensor.reshape(
                #         self.output_tensor.shape[1:]
                #     )
                # if len(self.output_tensor.shape) == 3:
                self.output_tensor = self.output_tensor.transpose(0, 2, 3, 1)
            elif len(self.output_tensor.shape) == 2:
                self.output_tensor = self.output_tensor.reshape(
                    1, 1, 1, self.output_tensor.size
                )
            self.order = "NWHC"

    def get_macs(self):
        return 0

    def flip_operation(self, flip):
        pass

    def __repr__(self):
        base_repr = "\nOperator name: {}\ttype: {}\tinput_layers: {}\toutput_shape: {}\n".format(
            self.name, self.type, self.input_layers, self.output_tensor.shape
        )
        return base_repr
