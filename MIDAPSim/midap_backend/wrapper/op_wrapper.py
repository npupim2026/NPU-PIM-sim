from midap_backend.wrapper.tensor_wrapper import TensorWrapper
from software.compiler.layer_compile import LayerInfo
from software.network.op_info import OpParams
from software.network.types import OpType

from .info import QuantInfo, TensorInfo


def convert_op_to_wrapper(layer_info: LayerInfo, tensor_dict=None):
    layer = layer_info.layer
    op: OpParams = layer.op
    type2wrapper = {
        OpType.Gemm: ConvWrapper,
        OpType.StdConv: ConvWrapper,
        OpType.MatMul: MatMulWrapper,
        OpType.MatMulTrans: ConvWrapper,
        OpType.Depthwise: DWConvWrapper,
        OpType.UpsampleBilinear: UpBilinearWrapper,
        OpType.AvgPool: AvgpoolWrapper,
        OpType.GlobalPool: AvgpoolWrapper,
        OpType.MaxPool: MaxpoolWrapper,
        OpType.Sum: AddWrapper,
        OpType.WeightedSum: SumWrapper,
        OpType.Mul: MulWrapper,
        OpType.RoPE: RoPEWrapper,
        OpType.HostProcess: HostProcessWrapper,
        OpType.Test: TestWrapper
    }
    if op.type not in type2wrapper:
        raise ValueError(f"Unknown op: {op}")
    kwargs = dict(
        name=op.name,
        activation=op.activation,
        output_data=layer.out_tensor.data
    )
    kwargs["kernel"] = op.kernel
    kwargs["pad"] = op.pad
    kwargs["stride"] = op.stride
    kwargs["dilation"] = op.dilation
    kwargs["broadcast"] = op.broadcast
    kwargs["in_plane"] = op.in_plane
    kwargs["global_pool"] = op.type == OpType.GlobalPool
    kwargs["macs"] = op.macs
    kwargs["test_code"] = op.test_code
    kwargs["mapping"] = op.mapping
    if layer.is_quantized:
        import math
        sign = 0
        main_shift = int(math.log2(layer.quant_info.scale_32b_to_16b))
        if main_shift < 0:
            sign = 1
            main_shift = -main_shift
        act_shift = -int(math.log2(layer.quant_info.scale_16b_to_8b))
        bias_shift = int(math.log2(layer.quant_info.bias_scale))
        lut = None if layer.quant_info.activation_lut is None else layer.quant_info.activation_lut.data
        qinfo = QuantInfo(sign, main_shift, act_shift, bias_shift, lut)
        kwargs["qinfo"] = qinfo
        if layer.quant_info.activation_lut is not None:
            kwargs["lut"] = layer.quant_info.activation_lut.name
    if op.weight is not None:
        kwargs["weight"] = op.weight.name
        kwargs["orig_weight_size"] = op.weight.init_size
        for i in range(len(tensor_dict)):
            tensor_dict[i][op.weight.name] = TensorInfo(op.weight.name, op.weight.data, True)
    elif op.type in [OpType.MatMul, OpType.MatMulTrans, OpType.RoPE]:
        weight = layer_info.ordered_input_vtensors[1].tensor
        kwargs["weight"] = weight.name
        kwargs["orig_weight_size"] = weight.init_size
    if op.bias is not None:
        kwargs["bias"] = op.bias.name
        for i in range(len(tensor_dict)):
            tensor_dict[i][op.bias.name] = TensorInfo(op.bias.name, op.bias.data, True)
    if len(layer.in_vtensors) > 1:
        in2_vtensor = layer_info.ordered_input_vtensors[1]
        tensor = TensorWrapper()
        tensor.set_tensor(
            name=in2_vtensor.name,
            shape=in2_vtensor.shape[1:],
            orig_shape=in2_vtensor.orig_shape[1:],
            init_shape=in2_vtensor.init_shape[1:],
            mapping_type=in2_vtensor.type,
            offset=in2_vtensor.offset,
            scale=in2_vtensor.scale,
            flip_x=in2_vtensor.flip_x,
        )
        kwargs["in2"] = tensor

    return type2wrapper[op.type](**kwargs)


class OpWrapper:
    def __init__(self, name, bias=None, lut = None, activation=None, qinfo=None, macs=0, **kwargs):
        self.name = name
        self.bias = bias
        self.activation = activation
        self.quant_info = qinfo  # QuantizationInfo
        self.lut = lut
        self.macs = macs

    def get_macs(self):
        return self.macs

    def __repr__(self):
        return self.name + '\n'


class ConvPoolWrapper(OpWrapper):
    def __init__(
        self,
        kernel=[1, 1],
        stride=1,
        pad=[0, 0, 0, 0],
        dilation=1,
        weight=None,
        orig_weight_size=0,
        in_plane=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kernel = kernel
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.weight = weight
        self.orig_weight_size = orig_weight_size
        self._in_plane = in_plane

    @property
    def k_h(self):
        return self.kernel[0]

    @property
    def k_w(self):
        return self.kernel[1]

    @property
    def pad_t(self):
        return self.pad[0]

    @property
    def pad_b(self):
        return self.pad[1]

    @property
    def pad_l(self):
        return self.pad[2]

    @property
    def pad_r(self):
        return self.pad[3]

    @property
    def in_plane(self):
        return self._in_plane

    def __repr__(self):
        rep_str = (
            "Name: {}, (kernel, stride, pad, dilation) = ({}, {}, {}, {})\n".format(
                self.name, self.kernel, self.stride, self.pad, self.dilation
            )
        )
        rep_str += "weight: {}, bias: {}\n".format(self.weight, self.bias)
        return rep_str


class DWWrapper(ConvPoolWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PoolWrapper(DWWrapper):
    def __init__(self, global_pool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_pool = global_pool

    def __repr__(self):
        return super().__repr__() + "global_pool: {}\n".format(self.global_pool)


class AvgpoolWrapper(PoolWrapper):
    def __repr__(self):
        return super().__repr__() + "Processing: Avgpool\n"


class MaxpoolWrapper(PoolWrapper):
    def __repr__(self):
        return super().__repr__() + "Processing: Maxpool\n"


class DWConvWrapper(DWWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return super().__repr__() + "Processing: Depthwise Convolution\n"


class UpBilinearWrapper(DWConvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ConvWrapper(ConvPoolWrapper):
    def __init__(self, group=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group = group  # Not in use


class MatMulWrapper(DWWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ArithmeticWrapper(OpWrapper):
    def __init__(self, in2=None, broadcast=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.broadcast = broadcast
        self.in2 = in2

    def __repr__(self):
        rep_str = "Name: {}, in2: {}, broadcast: {}\n".format(
            self.name, self.in2, self.broadcast
        )
        return rep_str


class AddWrapper(ArithmeticWrapper):
    pass


class MulWrapper(ArithmeticWrapper):
    pass

class SumWrapper(OpWrapper):
    def __init__(self, weight = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

class RoPEWrapper(DWWrapper):
    def __init__(self, weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

class HostProcessWrapper(OpWrapper):
    def __init__(self, output_data, mapping='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_data = output_data
        self.mapping = mapping

class TestWrapper(OpWrapper):
    def __init__(self, test_code=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_code = test_code
