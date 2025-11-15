import logging

from logger import init_logger
from software.compiler.layer_compile.layer_info import LayerInfo

from .info import ProcessingInfo, TensorInfo
from .op_wrapper import (
    ArithmeticWrapper,
    ConvPoolWrapper,
    RoPEWrapper,
    OpWrapper,
    convert_op_to_wrapper,
)
from .tensor_wrapper import TensorWrapper

logger = init_logger("Layer Wrapper", logging.DEBUG)


class LayerWrapper:
    def __init__(self):
        self.name = ""
        self.input_tensors = None
        self.main_op: OpWrapper = None
        self.main_output: TensorWrapper = None
        self.reduction_op: OpWrapper = None
        self.reduction_output: TensorWrapper = None
        self.processing_info: ProcessingInfo = (
            None  # Determine Processing type: Write, Load
        )
        # self.sync_info: SyncInfo = SyncInfo(-1, [])

    def from_layer_info(self, layer_info: LayerInfo, tensor_dict):
        self.name = layer_info.layer.name
        self.input_tensors = self.get_input_tensors(layer_info)
        self.set_processing_info(layer_info)
        self.set_op(layer_info, tensor_dict)
        # TODO Reduction Layer

    @property
    def input_tensor(self):
        return self.input_tensors[0] if self.input_tensors else None

    def get_input_tensors(self, layer_info: LayerInfo):
        # TODO: Virtualization
        # input tensor
        tensors = []
        for in_vtensor in layer_info.ordered_input_vtensors:
        # in_vtensor = layer_info.ordered_input_vtensors[0]
            input_tensor = TensorWrapper()
            input_tensor.set_tensor(
                name=in_vtensor.name,
                shape=in_vtensor.shape[1:],
                orig_shape=in_vtensor.orig_shape[1:],
                init_shape=in_vtensor.init_shape[1:],
                mapping_type=in_vtensor.type,
                offset=in_vtensor.offset,
                scale=in_vtensor.scale,
                flip_x=in_vtensor.flip_x,
            )
            tensors.append(input_tensor)
        # Input tensor for second input cannot be virtualized yet...
        return tensors

    def get_output_tensor(self, layer_info: LayerInfo):
        # TODO: Virtualization
        out_vtensor = layer_info.layer.out_vtensor
        # for upsampling, zero insertion
        tensor = TensorWrapper()
        tensor.set_tensor(
            name=out_vtensor.name,
            shape=out_vtensor.shape[1:],
            orig_shape=out_vtensor.orig_shape[1:],
            init_shape=out_vtensor.init_shape[1:],
            mapping_type=out_vtensor.type,
            offset=out_vtensor.offset,
            scale=out_vtensor.scale,
            flip_x=out_vtensor.flip_x,
        )
        return tensor

    def set_op(self, layer_info: LayerInfo, tensor_dict, reduction=False):
        output_vtensor = self.get_output_tensor(layer_info)
        pi = self.processing_info
        if output_vtensor.name not in tensor_dict:
            for i in range(len(tensor_dict)):
                data = layer_info.layer.out_vtensor.data[i, :]
                # if output_tensor.flip_x:
                #     data = np.flip(data, axis=0)
                tensor_dict[i][output_vtensor.name] = TensorInfo(
                    output_vtensor.name, data, False
                )
        # logger.debug(f"from {layer_info.layer}: {layer_info.layer.in_vtensors}, {layer_info.op}, {layer_info.out_vtensor}")
        wrapper_op = convert_op_to_wrapper(layer_info, tensor_dict)
        # logger.debug(f"to {wrapper_op.__dict__}")
        if isinstance(wrapper_op, ArithmeticWrapper):
            # TODO: how to determine in2 in Arithmetic operations?
            if wrapper_op.in2.name not in tensor_dict:
                for i in range(len(tensor_dict)):
                    tensor_dict[i][wrapper_op.in2.name] = TensorInfo(
                        wrapper_op.in2.name,
                        layer_info.ordered_input_tensors[-1].data[i, :],
                        False,
                    )
            pi.wmem_strategy.reverse_load = (
                self.input_tensor.flip_x != wrapper_op.in2.flip_x
            )
        elif isinstance(wrapper_op, RoPEWrapper):
            if wrapper_op.weight not in tensor_dict:
                for i in range(len(tensor_dict)):
                    tensor_dict[i][wrapper_op.weight] = TensorInfo(
                        wrapper_op.weight,
                        layer_info.ordered_input_tensors[-1].data[i, :],
                        False,
                    )
        if reduction:
            self.reduction_op = wrapper_op
            self.reduction_output = output_vtensor
        else:
            self.main_op = wrapper_op
            self.main_output = output_vtensor
            # pi.write_on_dram_pivot = min(pi.write_on_dram_pivot, pi.mapping_info.output_mapping[output_vtensor.name].write_on_dram_pivot)

    def set_processing_info(self, layer_info: LayerInfo):
        pi = ProcessingInfo()
        pi.mapping_info.add_layer_info(layer_info)
        # TODO : Add Reduction information
        # if layer_info.have_reduction_layer:
        #     pi.mapping_info.add_midap_layer_info(layer_info.next[0])
        pi.behavior_info.from_layer_info(layer_info, self.input_tensor)
        pi.reverse_write = layer_info.reversed  # TODO: Reverse Write
        self.processing_info = pi

    def __repr__(self):
        return self.name
