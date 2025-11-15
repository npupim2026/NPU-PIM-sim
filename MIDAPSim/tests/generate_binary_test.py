import argparse

import os
import sys

sys.path.append(os.path.dirname('./'))

from code_generator.opcode_tool import OpcodeTool
from config import cfg

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, required = True)
    parser.add_argument("-p", "--prefix", type=str, required = True)
    parser.add_argument("-od", "--output_dir", type=str, default = None)
    parser.add_argument("-op", "--output_prefix", type=str, default = None)
    parser.add_argument("-v", "--verify", action="store_true", default = False)
    parser.add_argument("-gen", "--generate", action="store_true", default = False)
    parser.add_argument("-itg", "--integrate", action="store_true", default = False)
    parser.add_argument("--debug", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    load_dir = args.dir
    prefix = args.prefix
    dram_file = None
    data_info_file = None
    config_file = os.path.join(load_dir, prefix + "_config.yml")
    inst_file = os.path.join(load_dir, prefix + "_inst.pkl")
    if args.verify or args.generate:
        mem_postfix = ["_dram_io.dat", "_dram_wb.dat", "_dram_buf.dat"]
        dram_file = []
        for postfix in mem_postfix:
            dram_file.append(os.path.join(load_dir, prefix + postfix))
        data_info_file = os.path.join(load_dir, prefix + "_data.pkl")
    kwargs = {}
    if args.output_dir is not None:
        kwargs['out_dir_name'] = args.output_dir
    if args.output_prefix is not None:
        kwargs['out_prefix'] = args.output_prefix
    if args.debug is not None:
        cfg.LOGGING_CONFIG_DICT["root"]["level"] = "DEBUG"
        cfg.LOGGING_CONFIG_DICT["root"]["handlers"] = ["console", "file"]
        cfg.LOGGING_CONFIG_DICT["loggers"]["debug"]["level"] = "DEBUG"
        kwargs['debug_layers'] = [args.debug]
    opcode_tool = OpcodeTool()
    opcode_tool.setup_generator(
            from_file = True,
            config_file = config_file,
            inst_file = inst_file,
            dram_file = dram_file,
            data_info_file = data_info_file
            )
    opcode_tool.run(
            verify = args.verify,
            generate_gold_data = args.generate,
            integrate = args.integrate,
            **kwargs
            )
