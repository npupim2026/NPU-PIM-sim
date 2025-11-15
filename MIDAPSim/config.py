from __future__ import absolute_import, division, print_function, unicode_literals

from data_structure.attrdict import AttrDict

__C = AttrDict()

cfg = __C

__C.SYSTEM = AttrDict()
# SYSTEM CONFIG : BANDWIDTH, DATASIZE(VIRTUAL), NETWORK

__C.SYSTEM.BANDWIDTH = 25.6  # GB ( * 10^9 byte) / s
__C.SYSTEM.DATA_SIZE = 1  # byte

__C.SYSTEM.FREQUENCY = 1000  # MHZ

# FIXME __C.SYSTEM.DRAM.PAGE_SIZE? other all DRAM related configurations..
# __C.SYSTEM.DRAM_PAGE_SIZE = 4 * 1024  # byte
# __C.SYSTEM.DRAM_PAGE_SIZE = 128  # byte

# LATENCY CONFIG : HOW MUCH CYCLE SPENT IN ..

__C.LATENCY = AttrDict()

__C.LATENCY.LATENCY_TYPE = "WORST"

__C.DRAM = AttrDict()  # 32bit channel * 2 LPDDR
__C.DRAM.COMM_TYPE = "VIRTUAL"  # VIRTUAL, TEST_DMA, DMA

__C.DRAM.FREQUENCY = 1.6  # GHz
__C.DRAM.INCLUDE_DRAM_WRITE = True
__C.DRAM.OFFSET = AttrDict()
__C.DRAM.OFFSET.SHARED = 0x00000000
__C.DRAM.OFFSET.WEIGHT_BIAS = 0x80000000
__C.DRAM.OFFSET.INSTRUCTION = 0x88000000
__C.DRAM.OFFSET.BUFFER = 0x90000000
__C.DRAM.OFFSET.INPUT = 0x98000000
__C.DRAM.OFFSET.OUTPUT = 0x9C000000
__C.DRAM.ADDRESS_ALIGN = 0x1000             # AXI protocol 4K boundary align
__C.DRAM.CLEAR_FRAME_MEM = False
# __C.DRAM.OFFSET.SPM = 0x90000000

##VIRTUAL DRAM Communication Constants
__C.DRAM.BUS_BANDWIDTH = 64
# Bus bandwidth
__C.DRAM.CHANNEL_SIZE = 4  # 32bit, 4byte
__C.DRAM.NUM_CHANNELS = 2
__C.DRAM.CAS = 35 / 1.6
__C.DRAM.PAGE_DELAY = 30 / 1.6
__C.DRAM.REFRESH_DELAY = 317 / 1.6
__C.DRAM.PAGE_SIZE = 8192
__C.DRAM.REFRESH_PERIOD = 3905  # Cycles
__C.DRAM.NUM_SLAVES = 2 # MIDAP Slaves

# TARGET ACCELERLATOR CONFIG
# MIDAP

__C.MODEL = AttrDict()

__C.MODEL.GENERATE_BLOCK_DOT = True
__C.MODEL.DOT_DIRECTORY = "./graph"

__C.MODEL.REDUCTION_LOGIC = True
__C.MODEL.USE_TILING = False
__C.MODEL.TILING_METHOD = None
__C.MODEL.TILING_OBJECTIVE = None

__C.MODEL.ALLOW_ABSTRACT_DATA = True
# True -> Memory mapping: to be supported in v1.3.0
# False -> Post module(subop): v1.2.0

__C.MODEL.QUANTIZED = False
# Quantized: 8bit integer fmap, weight / 32bit integer bias

__C.MODEL.NUM_FRAMES = 1

__C.MIDAP = AttrDict()

__C.MIDAP.EFFICENT_LOGIC = True
# Skip ineffective computation without delay

__C.MIDAP.SYSTEM_WIDTH = 64
__C.MIDAP.PACKET_SIZE = 1024
__C.MIDAP.FUNCTIONAL_SIM = True
__C.MIDAP.PLOT_DIRECTORY = "./simulation_plot"
__C.MIDAP.CORE_ID = 0

__C.MIDAP.FMEM = AttrDict()
__C.MIDAP.FMEM.NUM_ENTRIES = 256 * 1024  # # of Entries , PER ONE BANK
__C.MIDAP.FMEM.NUM = 4
__C.MIDAP.REDUCTION = AttrDict()
__C.MIDAP.REDUCTION.NUM_ENTRIES = 1024
__C.MIDAP.WRITE_BUFFER = AttrDict()
__C.MIDAP.WRITE_BUFFER.NUM_ENTRIES = 16 * 1024
# __C.MIDAP.FMEM.ALIGNMENT = 4 # Due to the Shift-Register limitation - maximum division - ex: 64, 4 --> 64 // 4 = 16 is a alignment channel size

__C.MIDAP.ALIGNMENT = 16
__C.MIDAP.WMEM = AttrDict()
__C.MIDAP.WMEM.USE_EXTENSION = False
__C.MIDAP.WMEM.NUM_ENTRIES = 9 * 1024  # # of Entries , PER ONE BANK
__C.MIDAP.WMEM.NUM = 16
__C.MIDAP.WMEM.E_NUM_ENTRIES = 16 * 1024
__C.MIDAP.BMEM = AttrDict()
__C.MIDAP.BMEM.NUM_ENTRIES = 2 * 1024  # Entries
__C.MIDAP.LUT = AttrDict()
__C.MIDAP.LUT.NUM_ENTRIES = 256  # Entries

# MIDAP CONTROL SEQUENCE GENERATOR STRATEGY
# It is a scheduling(mapping) problem!!
# There must be an optimal strategy, but it is hard to find without complex algorithm
# e. g. GA, ILP ....

__C.MIDAP.CONTROL_STRATEGY = AttrDict()

__C.MIDAP.CONTROL_STRATEGY.FIRST_LAYER = "GEMM"

__C.MIDAP.CONTROL_STRATEGY.FMEM = "INPUT_STATIONARY"
# INPUT_STATIONARY, OUTPUT_STATIONARY, GREEDY(will be supported)

# __C.MIDAP.CONTROL_STRATEGY.FILTER_LOAD = 'MAXIMUM'

__C.MIDAP.CONTROL_STRATEGY.FILTER_LOAD = "ONE"
# ONE, MAXIMUM, COMPILER_DRIVEN

__C.MIDAP.CONTROL_STRATEGY.WEIGHTED_SUM = False
__C.MIDAP.CONTROL_STRATEGY.LAYER_COMPILER = "MIN_DRAM_ACCESS"
__C.MIDAP.CONTROL_STRATEGY.GRAPH_TRANSFORMER = "PROTOTYPE"
__C.MIDAP.CONTROL_STRATEGY.MAPPING_POLICY = "LAYER_PIPE"
__C.MIDAP.CONTROL_STRATEGY.LOCAL_MEMORY_COMPILER = "PROTOTYPE"
__C.MIDAP.CONTROL_STRATEGY.L2_SPM_COMPILER = "PROTOTYPE"
# __C.MIDAP.CONTROL_STRATEGY.LAYER_COMPILER = 'HIDE_DRAM_LATENCY'

# __C.MIDAP.BUS_POLICY = 'FIFO'
__C.MIDAP.BUS_POLICY = "WMEM_FIRST"

# Backend

__C.BACKEND = AttrDict()
__C.BACKEND.CODE_SIZE = 4  # bytes
__C.BACKEND.REG_SIZE = 4  # bytes

# TODO: OTHER MIDAP CONFIGURATION OPTIONS WILL BE ADDED

__C.LOGGING_CONFIG_DICT = {
    "version": 1,
    "formatters": {
        "simple": {"format": "[%(name)s] %(message)s"},
        "complex": {
            "format": "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "error.log",
            "formatter": "simple",
            "level": "DEBUG",
        },
    },
    "root": {"handlers": ["console", "file"], "level": "INFO"},
    "loggers": {
        "debug": {"level": "DEBUG"},
        "gen": {"level": "DEBUG"},
        "op_sim": {"level": "DEBUG"},
    },
}
