from collections import OrderedDict

from midap_backend.wrapper.op_wrapper import HostProcessWrapper

from config import cfg
from data_structure.attrdict import from_attrdict_to_dict, from_dict_to_attrdict
from software.compiler.compile_info import CompileInfo
from software.compiler.layer_compile import LayerInfo
from software.network.op_info import OpType
from software.network.types import VTensorType
from software.system_compiler.core_info import CoreInfo

from .info import TensorInfo
from .layer_wrapper import LayerWrapper


class CompileWrapper:
    def __init__(self):
        self.layers: OrderedDict[str, LayerWrapper] = OrderedDict()
        self.tensor_dict = [OrderedDict() for i in range(cfg.MODEL.NUM_FRAMES)]  # tensor name: tensor
        self.config = None

    def from_core_info(self, core_info: CoreInfo):
        config = from_dict_to_attrdict(from_attrdict_to_dict(cfg))
        config.MIDAP.CORE_ID = core_info.core_id
        # config.MODEL.QUANTIZED = core_info.compile_info.layers[0].layer.is_quantized
        self.from_compile_info(core_info.compile_info, config)
        compile_info = core_info.compile_info
        for name, layer in self.layers.items():
            if isinstance(layer.main_op, HostProcessWrapper):
                continue
            pi = layer.processing_info
            pi.write_on_dram_pivot = core_info.local_memory_info.write_mem_pivot_dict[
                name
            ]
            pi.mapping_info.output_mapping[
                layer.main_output.name
            ].write_on_dram_pivot = pi.write_on_dram_pivot
            if name in compile_info.wmem_info_dict:
                wmem_info = compile_info.wmem_info_dict[layer.name]
                gs = wmem_info.filter_group_size
                lfo = wmem_info.load_filter_once
                pi.wmem_strategy.group_size = gs
                pi.wmem_strategy.load_filter_once = lfo
                pi.wmem_strategy.compute_type = wmem_info.compute_type
                pi.wmem_strategy.prepared = wmem_info.prepared
                pi.wmem_strategy.filter_name = wmem_info.filter_name
                if wmem_info.prepare_info is not None:
                    pi.wmem_strategy.prepare_info = self.layers[wmem_info.prepare_info]

    def from_compile_info(self, compile_info: CompileInfo, config=None):
        if config is None:
            self.config = from_dict_to_attrdict(
                from_attrdict_to_dict(cfg)
            )  ## Remove dependency
        else:
            self.config = config
        src_layers = compile_info.model.source
        for layer in src_layers:
            for in_vtensor in layer.in_vtensors:
                data_name = in_vtensor.name
                for i in range(len(self.tensor_dict)):
                    data = in_vtensor.data[i, :]
                    self.tensor_dict[i][data_name] = TensorInfo(
                        data_name, data, False, True
                    )  # It is not a fixed value..
        for layer, layer_info in compile_info.layer_dict.items():
            self.register_layer_info(layer_info)

    def register_layer_info(self, layer_info: LayerInfo):
        op_type = layer_info.op.type
        if op_type in [OpType.Concat, OpType.Upsample, OpType.Crop, OpType.Dummy]:
            pass
        elif layer_info.reduction:
            main = self.layers[layer_info.layer.inputs[0].name]
            main.set_op(layer_info, self.tensor_dict, reduction = True)
            main.processing_info.mapping_info.add_output_mapping(
                layer_info.layer.out_vtensor, layer_info.mapping.output
            )
        else:
            # Weight, bias
            layer_wrapper = LayerWrapper()
            layer_wrapper.from_layer_info(
                layer_info,
                self.tensor_dict,
            )
            if op_type in [OpType.MatMul, OpType.MatMulTrans] and (all([l.out_vtensor.type != VTensorType.OutputWMEM for l in layer_info.layer.parent.get_incoming_nodes(layer_info.layer)])):
                layer_wrapper.processing_info.wmem_strategy.reorder_load = True
            self.layers[layer_wrapper.name] = layer_wrapper
            return layer_wrapper
