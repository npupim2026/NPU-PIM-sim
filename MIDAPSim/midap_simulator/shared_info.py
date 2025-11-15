from typing import Any, Dict, List, Set

import numpy.typing as npt
import logging

from software.system_compiler.prefetch_info import TraceType, TransferTrace
from config import cfg

logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
logger = logging.getLogger('debug')

class SyncManager():
    __sync_layers : Set[Any] = set()

    @classmethod
    def init(cls):
        cls.__sync_layers = set()

    @classmethod
    def sync(cls, sync_id):
        cls.__sync_layers.add(sync_id)
    
    @classmethod
    def check_sync(cls, wait_id):
        return wait_id in cls.__sync_layers

    @classmethod
    def get_sync_layers(cls):
        return cls.__sync_layers

class PrefetchManager():
    __trace : List[List[TransferTrace]] = []
    @classmethod
    def get_trace(cls):
        return cls.__trace

    @classmethod
    def init_trace(cls, trace : Dict[int, List[TransferTrace]]):
        cls.__trace = list(trace.values())
    
    @classmethod
    def run_unit_transfer_single_core(cls, core_id : int, data : Dict[int, npt.ArrayLike]):
        # run maximum 1 transfer for each bank idx
        process_cnt = 0
        for idx in range(len(cls.__trace)):
            traces = cls.__trace[idx]
            while len(traces) > 0:
                trace = traces[0]
                if trace.type == TraceType.SYNC:
                    SyncManager.sync(trace.sync_id)
                elif trace.type == TraceType.WAIT:
                    sync_id = trace.sync_id
                    if not SyncManager.check_sync(sync_id):
                        # if sync_id[0] == core_id:
                        logger.debug(f"Bank {idx} Trace] Wait for {sync_id}...")
                        break
                        # else: # Ignore for single core
                        #     logger.info(f"Bank {idx} Trace] Sync wait failed: {sync_id} does not exist...")
                        #     logger.info(f"Ignore sync wait for different core")
                elif trace.type == TraceType.TRANSFER:
                    logger.debug(f"Bank {idx} Trace] Run transfer: {trace}")
                    sdt, saddr = trace.src
                    ddt, daddr = trace.dst
                    data[ddt][daddr:daddr + trace.size] = data[sdt][saddr:saddr + trace.size]
                    process_cnt += 1
                traces = traces[1:]
            cls.__trace[idx] = traces
        return process_cnt