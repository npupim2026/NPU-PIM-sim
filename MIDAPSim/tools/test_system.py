from __future__ import print_function
import os
import sys

sys.path.append(os.path.dirname("./"))

import argparse
from models.model_builder import ModelBuilder
from models.custom_torch.examples import densenet
import models.efficientnet.efficientnet as efficientnet
import models.examples as ex
import models.inception as inception
import models.mobilenet as mobilenet
import models.resnet as resnet
import models.se_resnet as se_resnet
import models.detector.yolo as yolov4
import models.detector.ssd as ssd
import models.segmentation.deeplab as deeplab
import models.segmentation.eff_unet as eff_unet
import models.yolo.yolov5.yolo as yolov5
import models.yolo.yolov7.yolo as yolov7
import models.yolo.yolov8.yolo as yolov8
import models.yolo.yolov9.yolo as yolov9
import models.test_model.examples as test_ex
import models.transformer.transformer_ex as transformer_ex
import models.transformer.vit as vit
import models.transformer.gemma as gemma
from config import cfg
from system_test_wrapper import SystemTestWrapper

def auto_int(x):
    return int(x, 0)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--network",
        type=str,
        choices=list(custom_examples.keys()) + ["all", "test"],
        required=True,
    )
    parser.add_argument("-i", "--input_size", type=int, default=0)
    parser.add_argument("-W", "--system_width", type=int, default=64)
    parser.add_argument("-N", "--num_cims", type=int, default=16)
    parser.add_argument(
        "-da", "--disable_abstract_layer", action="store_true", default=False
    )
    parser.add_argument(
        "-dr", "--disable_reduction_layer", action="store_true", default=False
    )
    parser.add_argument(
        "-de", "--disable_cim_extension", action="store_true", default=False
    )
    parser.add_argument(
        "-cf", "--clear_frame", action="store_true", default=False
    )
    parser.add_argument("-nf", "--num_frames", type=int, default=1)
    parser.add_argument(
        "-f",
        "--fmem_config",
        nargs=2,
        type=int,
        default=[8, 128],
        help="$num_banks $bank_size(KiB)",
    )
    parser.add_argument(
        "-w",
        "--wmem_config",
        nargs=3,
        type=int,
        default=[16, 9, 16],
        help="$num_cims $wmem_size(KiB) $ewmem_size(KiB)",
    )
    parser.add_argument("-sf", "--system_freq", type=float, default=500)
    parser.add_argument("-bb", "--bus_bandwidth", type=int, default=128)
    parser.add_argument("-df", "--dram_freq", type=float, default=1.6)
    parser.add_argument("-dc", "--dram_chan", type=int, default=2)
    parser.add_argument("--level", type=int, default=0, help="Simulation Level")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "-d",
        "--dram_comm_type",
        choices=["DMA", "TEST_DMA", "VIRTUAL", "TEST_3D"],
        default="VIRTUAL",
    )
    parser.add_argument(
        "-dws",
        "--disable_weighted_sum",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-fl",
        "--first_layer",
        choices=["GEMM", "PAD", "EXCLUDE"],
        default="GEMM",
    )
    parser.add_argument( # Add graph transformer options
        "-gt",
        "--graph_transformer",
        choices=["PROTOTYPE"],
        default="PROTOTYPE"
    )
    parser.add_argument( # Add mapping policy options
        "-mp",
        "--mapping_policy",
        choices=["PROTOTYPE", "LAYER_PIPE", 'HOST'],
        default="LAYER_PIPE"
    )
    parser.add_argument( # Add local memory compiler options
        "-lmc",
        "--local_memory_compiler",
        choices=["PROTOTYPE"],
        default="PROTOTYPE"
    )
    parser.add_argument( # Add l2 spm compiler options
        "-lsc",
        "--l2_spm_compiler",
        choices=["PROTOTYPE","N2N"],
        default="PROTOTYPE"
    )
    parser.add_argument(
        "--spm_config",
        nargs=2,
        type=int,
        default=[0, 0],
        help="$(num_banks) $(bank_size(KiB))",
    )
    parser.add_argument("-nc", "--num_cores", type=int, default=1)
    parser.add_argument("-ps", "--packet_size", type=int, default=1024)
    parser.add_argument(
        "-mo",
        "--mem_offset",
        nargs=5,
        type=auto_int,
        default=[0, 0x98000000, 0x9C000000, 0x80000000, 0x90000000],
        help="$io_offset $const_offset $buffer(spm)_offset",
    )
    parser.add_argument(
        "-fs", "--functional_simulation", action="store_true", default=False
    )
    parser.add_argument("-q", "--quantize", action="store_true", default=False)
    parser.add_argument("-so", "--save_only", action="store_true", default=False)
    parser.add_argument("-sd", "--save_dir", type=str, default=None)
    parser.add_argument("-sp", "--save_prefix", type=str, default=None)
    parser.add_argument("-idx", "--core_idx", type=int, default=1)
    return parser.parse_args()


custom_examples = {
    "dcgan": (lambda x: ex.dcgan(nz=x[-1]) if x is not None else ex.dcgan()),
    "discogan": (
        lambda x: ex.discogan(input_shape=x) if x is not None else ex.discogan()
    ),
    "unet": (
        lambda x: ex.unet(input_shape=x, decompose=True)
        if x is not None
        else ex.unet(decompose=True)
    ),
    "unet_small": (
        lambda x: ex.unet_small(input_shape=x, decompose=not args.disable_abstract_layer)
        if x is not None
        else ex.unet_small(decompose=not args.disable_abstract_layer)
    ),
    "test_network": (
        lambda x: ex.test_network(x) if x is not None else ex.test_network()
    ),
    "test_dilation": (
        lambda x: test_ex.dilation_test(x) if x is not None else test_ex.dilation_test()
    ),
    "test_bilinear": (
        lambda x: test_ex.bilinear_test(x) if x is not None else test_ex.bilinear_test()
    ),
    "resnet50": (
        lambda x: resnet.resnet50(input_size=x[-1])
        if x is not None
        else resnet.resnet50()
    ),
    "resnet101": (
        lambda x: resnet.resnet101(input_size=x[-1])
        if x is not None
        else resnet.resnet101()
    ),
    "resnet152": (
        lambda x: resnet.resnet152(input_size=x[-1])
        if x is not None
        else resnet.resnet152()
    ),
    "inception_v3": (
        lambda x: inception.inception_v3(input_size=x[-1])
        if x is not None
        else inception.inception_v3()
    ),
    "se_resnet50": (
        lambda x: se_resnet.se_resnet50(input_size=x[-1])
        if x is not None
        else se_resnet.se_resnet50()
    ),
    "se_resnet101": (
        lambda x: se_resnet.se_resnet101(input_size=x[-1])
        if x is not None
        else se_resnet.se_resnet101()
    ),
    "se_resnet152": (
        lambda x: se_resnet.se_resnet152(input_size=x[-1])
        if x is not None
        else se_resnet.se_resnet152()
    ),
    "mobilenet": (
        lambda x: mobilenet.mobilenet(input_size=x[-1])
        if x is not None
        else mobilenet.mobilenet()
    ),
    "mobilenet_v2": (
        lambda x: mobilenet.mobilenet_v2(input_size=x[-1])
        if x is not None
        else mobilenet.mobilenet_v2()
    ),
    'mobilenet_v3_small': (
        lambda x: mobilenet.mobilenet_v3(input_size=x[-1])
        if x is not None
        else mobilenet.mobilenet_v3()
    ),
    'mobilenet_v3_small_0.75': (
        lambda x: mobilenet.mobilenet_v3(input_size=x[-1], depth_multiplier=0.75)
        if x is not None
        else mobilenet.mobilenet_v3(depth_multiplier=0.75)
    ),
    'mobilenet_v3_small_minimal': (
        lambda x: mobilenet.mobilenet_v3_small_minimal(input_size=x[-1])
        if x is not None
        else mobilenet.mobilenet_v3_small_minimal()
    ),
    'mobilenet_v3_large': (
        lambda x: mobilenet.mobilenet_v3_large(input_size=x[-1])
        if x is not None
        else mobilenet.mobilenet_v3_large()
    ),
    'mobilenet_v3_large_0.75': (
        lambda x: mobilenet.mobilenet_v3_large(input_size=x[-1], depth_multiplier=0.75)
        if x is not None
        else mobilenet.mobilenet_v3_large(depth_multiplier=0.75)
    ),
    'mobilenet_v3_large_1.25': (
        lambda x: mobilenet.mobilenet_v3_large(input_size=x[-1], depth_multiplier=1.25)
        if x is not None
        else mobilenet.mobilenet_v3_large(depth_multiplier=1.25)
    ),
    'mobilenet_v3_large_minimal': (
        lambda x: mobilenet.mobilenet_v3_large_minimal(input_size=x[-1])
        if x is not None
        else mobilenet.mobilenet_v3_large_minimal()
    ),
    'mobilenet_v3_edgetpu': (
        lambda x: mobilenet.mobilenet_v3_edgetpu(input_size=x[-1])
        if x is not None
        else mobilenet.mobilenet_v3_edgetpu()
    ),
    'mobilenet_v3_edgetpu_0.75': (
        lambda x: mobilenet.mobilenet_v3_edgetpu(input_size=x[-1], depth_multiplier=0.75)
        if x is not None
        else mobilenet.mobilenet_v3_edgetpu(depth_multiplier=0.75)
    ),
    'mobilenet_v3_edgetpu_1.25': (
        lambda x: mobilenet.mobilenet_v3_edgetpu(input_size=x[-1], depth_multiplier=1.25)
        if x is not None
        else mobilenet.mobilenet_v3_edgetpu(depth_multiplier=1.25)
    ),
    "mobilenet_ssd": (
        lambda x: ssd.mobilenet_ssd(input_size=x[-1])
        if x is not None
        else ssd.mobilenet_ssd()
    ),
    "yolov4_tiny": (
        lambda x: yolov4.yolov4_tiny(input_size=x[-1])
        if x is not None
        else yolov4.yolov4_tiny(input_size=416)
    ),
    "wide_resnet101": (
        lambda x: resnet.wide_resnet101_2(input_size=x[-1])
        if x is not None
        else resnet.wide_resnet101_2()
    ),
    "wide_resnet50": (
        lambda x: resnet.wide_resnet50_2(input_size=x[-1])
        if x is not None
        else resnet.wide_resnet50_2()
    ),
    "efficientnet-b0": (
        lambda x: efficientnet.efficientnet(
            "efficientnet-b0", input_channels=x[-1] if x is not None else None
        )
    ),
    "efficientnet-b1": (
        lambda x: efficientnet.efficientnet(
            "efficientnet-b1", input_channels=x[-1] if x is not None else None
        )
    ),
    "efficientnet-b2": (
        lambda x: efficientnet.efficientnet(
            "efficientnet-b2", input_channels=x[-1] if x is not None else None
        )
    ),
    "efficientnet-b3": (
        lambda x: efficientnet.efficientnet(
            "efficientnet-b3", input_channels=x[-1] if x is not None else None
        )
    ),
    "efficientnet-b4": (
        lambda x: efficientnet.efficientnet(
            "efficientnet-b4", input_channels=x[-1] if x is not None else None
        )
    ),
    "efficientnet-b5": (
        lambda x: efficientnet.efficientnet(
            "efficientnet-b5", input_channels=x[-1] if x is not None else None
        )
    ),
    "efficientnet-b6": (
        lambda x: efficientnet.efficientnet(
            "efficientnet-b6", input_channels=x[-1] if x is not None else None
        )
    ),
    "efficientnet-b7": (
        lambda x: efficientnet.efficientnet(
            "efficientnet-b7", input_channels=x[-1] if x is not None else None
        )
    ),
    #    'efficientnet-b8'  : (lambda x: efficientnet.efficientnet('efficientnet-b8', input_channels=x[-1] if x is not None else None)),
    #    'efficientnet-l2'  : (lambda x: efficientnet.efficientnet('efficientnet-l2', input_channels=x[-1] if x is not None else None)),
    "densenet121": (
        lambda x: densenet.get_densenet(
            blocks=121,
            model_name="densenet121",
            input_size=x[-1] if x is not None else None,
        )
    ),
    "densenet161": (
        lambda x: densenet.get_densenet(
            blocks=161,
            model_name="densenet161",
            input_size=x[-1] if x is not None else None,
        )
    ),
    "densenet169": (
        lambda x: densenet.get_densenet(
            blocks=169,
            model_name="densenet169",
            input_size=x[-1] if x is not None else None,
        )
    ),
    "densenet201": (
        lambda x: densenet.get_densenet(
            blocks=201,
            model_name="densenet201",
            input_size=x[-1] if x is not None else None,
        )
    ),
    "deeplabv3+mv": (
        lambda x: deeplab.deeplab_v3(
            backbone='mobilenet',
            input_size=x[-1] if x is not None else 320
        )
    ),
    "eff_unet": (
        lambda x: eff_unet.eff_unet(
            net='efficientnet-b0',
            input_size=x[-1] if x is not None else 256,
            out_channels=2,
            decompose = not args.disable_abstract_layer
        )
    ),
    "yolov5s": (
        lambda x: yolov5.yolov5s(
            input_size=x[-1] if x is not None else 640,
        )
    ),
    "yolov5m": (
        lambda x: yolov5.yolov5m(
            input_size=x[-1] if x is not None else 640,
        )
    ),
    "yolov5l": (
        lambda x: yolov5.yolov5l(
            input_size=x[-1] if x is not None else 640,
        )
    ),
    "yolov7": (
        lambda x: yolov7.yolov7(
            input_size=x[-1] if x is not None else 640,
        )
    ),
    "yolov7_tiny": (
        lambda x: yolov7.yolov7_tiny(
            input_size=x[-1] if x is not None else 640,
        )
    ),
    "yolov8s": (
        lambda x: yolov8.yolov8s(
            input_size=x[-1] if x is not None else 640,
        )
    ),
    "yolov8m": (
        lambda x: yolov8.yolov8m(
            input_size=x[-1] if x is not None else 640,
        )
    ),
    "yolov8l": (
        lambda x: yolov8.yolov8l(
            input_size=x[-1] if x is not None else 640,
        )
    ),
    "yolov9": (
        lambda x: yolov9.yolov9(
            input_size=x[-1] if x is not None else 640,
        )
    ),
    "yolov9-c": (
        lambda x: yolov9.yolov9_c(
            input_size=x[-1] if x is not None else 640,
        ),
    ),
    #"attention": (
    #    lambda x: transformer_ex.attention_head()
    #),
    #"matmul_test": (
    #    lambda x: transformer_ex.matmul_test()
    #),
    #"mha": (
    #    lambda x: transformer_ex.multi_head_attention_test()
    #),
    #"vit_block": (
    #    lambda x: transformer_ex.vit_block_test()
    #),
    "vit": (
        lambda x: vit.ViT(image_size=x[-1])
        if x is not None
        else vit.ViT()
    ),
    "vit_layer": (
        lambda x: vit.ViTLayer(image_size=x[-1])
        if x is not None
        else vit.ViTLayer()
    ),
    "gemma2": (
        lambda x: gemma.Gemma2(input_len=x[-1])
        if x is not None
        else gemma.Gemma2()
    ),
    "gemma2-2b": (
        lambda x: gemma.Gemma2_2b(input_len=x[-1])
        if x is not None
        else gemma.Gemma2_2b()
    ),
    "gemma2_layer": (
        lambda x: gemma.GemmaLayer(input_len=x[-1])
        if x is not None
        else gemma.GemmaLayer()
    ),
    "gemma2_layer_pim_offload": (
        lambda x: gemma.GemmaLayer_pim_offload(input_len=x[-1])
        if x is not None
        else gemma.GemmaLayer_pim_offload()
    ),
    "gemma2-2b_pim_offload": (
        lambda x: gemma.Gemma2_2b_pim_offload(input_len=x[-1])
        if x is not None
        else gemma.Gemma2_2b_pim_offload()
    ),
}

test_set = [
    # "discogan",
    "resnet50",
    # "wide_resnet50",
    # "inception_v3",
    # "se_resnet50",
    "mobilenet_v2",
    # "efficientnet-b3",
]

if __name__ == "__main__":
    args = parse()
    # Architectural Configuration
    cfg.MIDAP.SYSTEM_WIDTH = args.system_width
    cfg.MIDAP.FMEM.NUM = args.fmem_config[0]
    cfg.MIDAP.FMEM.NUM_ENTRIES = args.fmem_config[1] * 1024
    cfg.MIDAP.WMEM.NUM = args.wmem_config[0]
    cfg.MIDAP.WMEM.NUM_ENTRIES = args.wmem_config[1] * 1024
    cfg.MIDAP.WMEM.E_NUM_ENTRIES = args.wmem_config[2] * 1024
    cfg.MIDAP.CONTROL_STRATEGY.WEIGHTED_SUM = not args.disable_weighted_sum
    ## TODO: System Compiler options
    cfg.MIDAP.CONTROL_STRATEGY.GRAPH_TRANSFORMER = args.graph_transformer
    cfg.MIDAP.CONTROL_STRATEGY.MAPPING_POLICY = args.mapping_policy
    cfg.MIDAP.CONTROL_STRATEGY.LOCAL_MEMORY_COMPILER = args.local_memory_compiler
    cfg.MIDAP.CONTROL_STRATEGY.L2_SPM_COMPILER = args.l2_spm_compiler 
    ## TODO : Core Compiler options
    ## TODO: Core Compiler
    cfg.MIDAP.CONTROL_STRATEGY.FIRST_LAYER = args.first_layer
    cfg.MODEL.ALLOW_ABSTRACT_DATA = not args.disable_abstract_layer
    cfg.MODEL.REDUCTION_LOGIC = not args.disable_reduction_layer
    cfg.MIDAP.WMEM.USE_EXTENSION = not args.disable_cim_extension
    cfg.MODEL.NUM_FRAMES = args.num_frames

    ## Simulation configurations
    cfg.SYSTEM.FREQUENCY = args.system_freq
    cfg.DRAM.CLEAR_FRAME_MEM = args.clear_frame
    cfg.DRAM.BUS_BANDWIDTH = args.bus_bandwidth
    cfg.DRAM.FREQUENCY = args.dram_freq
    cfg.DRAM.NUM_CHANNELS = args.dram_chan
    cfg.DRAM.PAGE_SIZE = 4096 * args.dram_chan
    cfg.DRAM.COMM_TYPE = args.dram_comm_type
    cfg.DRAM.OFFSET.SHARED = args.mem_offset[0]
    cfg.DRAM.OFFSET.INPUT = args.mem_offset[1]
    cfg.DRAM.OFFSET.OUTPUT = args.mem_offset[2]
    cfg.DRAM.OFFSET.WEIGHT_BIAS = args.mem_offset[3]
    cfg.DRAM.OFFSET.BUFFER = args.mem_offset[4]
    cfg.SYSTEM.BANDWIDTH = (
        cfg.DRAM.CHANNEL_SIZE * cfg.DRAM.FREQUENCY * cfg.DRAM.NUM_CHANNELS * 2
    ) / cfg.SYSTEM.DATA_SIZE

    cfg.MIDAP.FUNCTIONAL_SIM = args.functional_simulation
    if args.quantize == False:
        cfg.MIDAP.PACKET_SIZE = int(args.packet_size // 4)
    else:
        cfg.MIDAP.PACKET_SIZE = int(args.packet_size)
    #   deprecated
    #   if cfg.MIDAP.CORE_ID >= 0:
    #       cfg.DRAM.DUMP_FILE = os.path.dirname(os.path.realpath(__file__)) + str("/../shared/.args.dram" + str(cfg.MIDAP.CORE_ID) + ".dat")

    # Debug information
    if args.debug:
        cfg.LOGGING_CONFIG_DICT["root"]["level"] = "DEBUG"
        cfg.LOGGING_CONFIG_DICT["root"]["handlers"] = ["console", "file"]
        cfg.LOGGING_CONFIG_DICT["loggers"]["debug"]["level"] = "DEBUG"

    input_shape = (
        None if args.input_size == 0 else (1, 3, args.input_size, args.input_size)
    )

    def _run(tr: SystemTestWrapper, mb: ModelBuilder, last: bool = True):
        save = False
        kwargs = {
            "quantize": args.quantize,
            "spm_num_banks": args.spm_config[0],
            "spm_bank_size": args.spm_config[1] * 1024,
            "num_frames": args.num_frames,
        }
        if args.save_only or args.save_dir or args.save_prefix is not None:
            save = True
            kwargs["save_dir"] = "./temp" if args.save_dir is None else args.save_dir
            kwargs["prefix"] = mb.name if args.save_prefix is None else args.save_prefix
        kwargs["sim_last"] = last
        return tr.run_all(mb, run=not args.save_only, save=save, **kwargs)

    functionality_check = {}
    if args.network == "all":
        for i, model in enumerate(custom_examples):
            tr = SystemTestWrapper(args.num_cores, args.core_idx, args.level)
            mb = custom_examples[model](input_shape)
            functionality_check[model] = _run(tr, mb, i == len(custom_examples) - 1)
    elif args.network == "test":
        for i, network in enumerate(test_set):
            tr = SystemTestWrapper(args.num_cores, args.core_idx, args.level)
            mb = custom_examples[network](input_shape)
            functionality_check[network] = _run(tr, mb, i == len(test_set) - 1)
    elif args.network in custom_examples:
        tr = SystemTestWrapper(args.num_cores, args.core_idx, args.level)
        mb = custom_examples[args.network](input_shape)
        functionality_check[args.network] = _run(tr, mb)
    else:
        raise ValueError(
            "{} is not supported model.. please check custom_examples list.".format(
                args.network
            )
        )
    if cfg.MIDAP.FUNCTIONAL_SIM:
        import logging
        logger = logging.getLogger()
        logger.info("Functionality Check Result:")
        for result in functionality_check:
            for i, msg in enumerate(functionality_check[result]):
                logger.info(f"    {result}_core_{i + 1}: {msg}")
