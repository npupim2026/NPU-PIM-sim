import argparse
from system_test_wrapper import SystemTestWrapper
from config import cfg

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, required = True)
    parser.add_argument("-p", "--prefix", type=str, required = True)
    parser.add_argument(
        "-id", "--core_id",
        type = int,
        default = -1,
        help = "-1: System simulation(virtual), else: run core only")
    parser.add_argument(
        "-idx", "--core_idx",
        type = int,
        default = 1)
    parser.add_argument("-l", "--simulation_level", type=int, default = 0)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    if args.debug:
        cfg.LOGGING_CONFIG_DICT["root"]["level"] = "DEBUG"
        cfg.LOGGING_CONFIG_DICT["root"]["handlers"] = ["console", "file"]
        cfg.LOGGING_CONFIG_DICT["loggers"]["debug"]["level"] = "DEBUG"
    loader = SystemTestWrapper(simulation_level=args.simulation_level, core_idx=args.core_idx)
    if args.core_id == -1:
        loader.run_system_from_dir(args.dir, args.prefix)
    else:
        import os
        core_dir = os.path.join(args.dir, args.prefix, f"core_{args.core_id}")
        loader.run_core_from_dir(core_dir)

