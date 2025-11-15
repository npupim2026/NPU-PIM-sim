import argparse

import os
import sys

sys.path.append(os.path.dirname('./'))

from code_generator.opcode_tool import OpcodeTool
from config import cfg
from system_test_wrapper import SystemTestWrapper

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
    parser.add_argument("-da", "--dynamic_addr", action="store_true", default = False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    load_dir = os.path.join(args.dir, args.prefix)
    opcode_tools = [] # Debugging info
    if args.debug is not None:
        cfg.LOGGING_CONFIG_DICT["root"]["level"] = "DEBUG"
        cfg.LOGGING_CONFIG_DICT["root"]["handlers"] = ["console", "file"]
        cfg.LOGGING_CONFIG_DICT["loggers"]["debug"]["level"] = "DEBUG"
    def run_func(core_dir : str, mem_data):
        dir_name = core_dir.split(os.path.sep)[-1]
        config_file = os.path.join(core_dir, "config.yml")
        inst_file = os.path.join(core_dir, "inst.pkl")        
        data_info_file = None
        if args.verify or args.generate:
            data_info_file = os.path.join(core_dir, "gt_info.pkl")
        opcode_tool = OpcodeTool()
        opcode_tools.append(opcode_tool)
        kwargs = {}
        if args.output_dir is not None:
            kwargs['out_dir_name'] = os.path.join(args.output_dir)
        if args.output_prefix is not None:
            kwargs['out_prefix'] = os.path.join(args.output_prefix, dir_name)
        if args.debug is not None:
            kwargs['debug_layers'] = [args.debug]
        opcode_tool.setup_generator(
                from_file = True,
                config_file = config_file,
                inst_file = inst_file,
                dram_data = mem_data,
                data_info_file = data_info_file,
                dynamic_base_addr = args.dynamic_addr
                )
        opcode_tool.run(
                verify = args.verify,
                generate_gold_data = args.generate,
                integrate = args.integrate,
                **kwargs
                )
    SystemTestWrapper.run_system_func(load_dir, run_func)
