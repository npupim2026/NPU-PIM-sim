from __future__ import annotations

import logging
from enum import Enum, auto
from typing import TYPE_CHECKING

import attr
from logger import init_logger

if TYPE_CHECKING:
    from typing import Dict, Tuple

__author__ = "Donghyun Kang"
__copyright__ = "Copyright 2020, MIDAP Project"
__credits__ = ["Donghyun Kang"]

__license__ = "MIT License"
__maintainer__ = "Donghyun Kang"
__status__ = "Production"

logger = init_logger("Prefetch Info", logging.INFO)


class TraceType(Enum):
    SYNC = auto()
    WAIT = auto()
    TRANSFER = auto()


@attr.s(slots=True, init=True)
class PrefetchInfo(object):
    name: str = attr.ib()
    src: str = attr.ib()
    dst: str = attr.ib()
    size: int = attr.ib()


@attr.s(slots=True, init=True, repr=False)
class TransferTrace(object):
    type: TraceType = attr.ib()
    sync_id: Tuple[int, int] = attr.ib(default=None)
    src: Tuple[int, int] = attr.ib(default=None)
    dst: Tuple[int, int] = attr.ib(default=None)
    size: int = attr.ib(default=None)

    @classmethod
    def Transfer(cls, src, dst, size):
        return cls(type=TraceType.TRANSFER, src=src, dst=dst, size=size)

    @classmethod
    def Sync(cls, id):
        return cls(type=TraceType.SYNC, sync_id=id)

    @classmethod
    def Wait(cls, id):
        return cls(type=TraceType.WAIT, sync_id=id)

    def to_str(self, offset_info: Dict[int, int]):
        if self.type == TraceType.TRANSFER:
            src_addr = offset_info[self.src[0]] + self.src[1]
            dst_addr = offset_info[self.dst[0]] + self.dst[1]
            return f"TRANSFER {src_addr} {dst_addr} {self.size}\n"
        if self.type == TraceType.WAIT:
            return f"WAIT {self.sync_id[0]} {self.sync_id[1]}\n"
        if self.type == TraceType.SYNC:
            return f"SYNC {self.sync_id[0]} {self.sync_id[1]}\n"

    def __repr__(self):
        if self.type == TraceType.TRANSFER:
            return f"TRANSFER {self.src} {self.dst} {self.size}\n"
        if self.type == TraceType.WAIT:
            return f"WAIT {self.sync_id[0]} {self.sync_id[1]}\n"
        if self.type == TraceType.SYNC:
            return f"SYNC {self.sync_id[0]} {self.sync_id[1]}\n"