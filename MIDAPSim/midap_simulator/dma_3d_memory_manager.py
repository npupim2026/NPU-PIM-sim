from __future__ import absolute_import, division, print_function, unicode_literals

import math
import logging

import numpy as np
from collections import deque
from functools import reduce

from .dma_memory_manager import TDMemoryManager, RequestInfo, QueueElement
from config import cfg

'''
    3D DMA version for the MemoryManager class
'''

class Descriptor3D():
    next_id = 0

    def __init__(
        self, aoff, acnt, boff, bcnt, coff, ccnt,
        next_dsc: int = 0,
        csub: bool = False,
        bsub: bool = False,
        bcsync: bool = False,
        intenb: bool = False,
        last: bool = False,
        baddr_ddr: int = 0,
        baddr_sram = None
    ):
        self.id = Descriptor3D.get_id()
        self.aoff = int(aoff)
        self.acnt = int(acnt)
        self.boff = int(boff)
        self.bcnt = int(bcnt)
        self.coff = int(coff)
        self.ccnt = int(ccnt) 

        self.next_dsc = int(next_dsc)
        self.csub = csub
        self.bsub = bsub
        self.bcsync = bcsync
        self.intenb = intenb
        self.last = last
        self.baddr_ddr = int(baddr_ddr)
        self.baddr_sram = baddr_sram

    @classmethod
    def get_id(cls):
        result = cls.next_id
        cls.next_id += 1
        return result

class DMA3DMemoryManager(TDMemoryManager):
    def __init__(self, manager):
        super().__init__(manager)
        self.tmem = np.reshape(self.tmem, -1)                       # No external tmem; write buffer is in the DMA
        self.buffer_write_pivot = 0                                 # Pivot for the circular buffer
        self.buffer_read_pivot = 0                                  # Pivot for the circular buffer
        self.queue_crits = [self.id_offsets[1], self.id_offsets[1]] # FIXME: Temporal solution - No priority
        self.rdma_des_queue = deque()                               # Implemented with a queue instead of a table for simplicity
        self.wdma_des_queue = deque()
        self.wdma_wait_list = deque()
        self.rdma_timer = 0
        self.wdma_timer = 0
        self.rdma_next_reset = 0    # Weird but worth to adopt for simplicity
        self.wdma_next_reset = 0
        self.write_available_space = self.tmem.size
        self.wdma_enabled = False
        self.write_dram_addr = -1
        self.aiter = 0
        self.biter = 0
        self.citer = 0
        self.dma_setup_delay = 370  # Based on the experimental result in RSP2023

    def reset_sram(self):
        self.wmem_in_use = -1
        self.bmem_in_use = -1
        self.fmem[:, :] = 0
        self.wmem[:, :, :] = 0
        self.bmem[:, :] = 0
        self.tmem[:] = 0
        if self.lut_raw is not None:
            self.lut_raw[:] = 0

    def register_wdma_descriptor(self, descriptor_list: [Descriptor3D]):
        self.wdma_des_queue.extend(descriptor_list)

    def add_request_from_rdesc(self, mem_id, descriptor: Descriptor3D):
        base_sram = descriptor.baddr_sram
        base_dram = descriptor.baddr_ddr
        acnt = descriptor.acnt
        aoff = descriptor.aoff
        boff = descriptor.boff * ((-1) ** descriptor.bsub)
        coff = descriptor.coff * ((-1) ** descriptor.csub) + int(not descriptor.bcsync) * boff * descriptor.bcnt
        request_id = self.add_request(  # DMA setup delay
            mem_id=-1,
            request_type=2,
            request_size=0,
            buf=None,
            offset=0,
            dram_address=0
        )
        for c in range(descriptor.ccnt):
            for b in range(descriptor.bcnt):
                request_id = self.add_request(
                    mem_id=mem_id,
                    request_type=0,
                    request_size=acnt,
                    buf=base_sram,
                    offset=aoff + b * boff + c * coff,
                    dram_address=base_dram + (b * c + b) * acnt
                )
        return request_id

    def load_wmem(self, to_all, filter_name, filter_size = 0, filter_offset = 0, wmem_offset = 0, dram_offset = 0, continuous_request = False):
        wmem_not_in_use = (self.wmem_in_use + 1) % 2
        dram_address = self.get_dram_address(filter_name, dram_offset)
        if filter_size == filter_offset or not to_all:
            descriptor = Descriptor3D(
                wmem_offset, filter_size, self.ewmem_size, self.num_wmem if to_all else 1, filter_size, 1,
                intenb=True, last=True, baddr_ddr=dram_address, baddr_sram=self.wmem[wmem_not_in_use].view().reshape(-1)
            )
            request_id = self.add_request_from_rdesc(self.wid_offset + wmem_not_in_use, descriptor)
            if self.rdma_des_queue:
                self.rdma_des_queue[-1].last = False
                self.rdma_des_queue[-1].next_dsc = descriptor.id
            self.rdma_des_queue.append(descriptor)  # for debugging...?
            self.wait_request_info[self.wid_offset + wmem_not_in_use] = request_id
        else:   # The strided pattern in DRAM cannot be handled...? (dram_offset)
            descriptors = [Descriptor3D(
                wmem_offset, filter_size, self.ewmem_size, 1, filter_size, 1,
                intenb=True, last=True, baddr_ddr=dram_address+i*filter_offset, baddr_sram=self.wmem[wmem_not_in_use][i]
            ) for i in range(self.num_wmem)]
            for descriptor in descriptors:
                request_id = self.add_request_from_rdesc(self.wid_offset + wmem_not_in_use, descriptor)
                if self.rdma_des_queue:
                    self.rdma_des_queue[-1].last = False
                    self.rdma_des_queue[-1].next_dsc = descriptor.id
                self.rdma_des_queue.append(descriptor)
                self.wait_request_info[self.wid_offset + wmem_not_in_use] = request_id

    def load_fmem(self, fmem_idx, data_name, data_size, fmem_offset = 0, dram_offset = 0):
        dram_address = self.get_dram_address(data_name, dram_offset)
        mem_id = self.fid_offset + fmem_idx
        descriptor = Descriptor3D(
            fmem_offset, data_size, data_size, 1, data_size, 1,
            intenb=True, last=True, baddr_ddr=dram_address, baddr_sram=self.fmem[fmem_idx]
        )
        request_id = self.add_request_from_rdesc(mem_id, descriptor)
        if self.rdma_des_queue:
            self.rdma_des_queue[-1].last = False
            self.rdma_des_queue[-1].next_dsc = descriptor.id
        self.rdma_des_queue.append(descriptor)
        self.wait_request_info[mem_id] = request_id

    def load_bmem(self, bias_name, bias_size):
        bmem_not_in_use = (self.bmem_in_use + 1) % 2
        dram_address = self.get_dram_address(bias_name, 0)
        mem_id = self.bid_offset + bmem_not_in_use
        descriptor = Descriptor3D(
            0, bias_size, bias_size, 1, bias_size, 1,
            intenb=True, last=True, baddr_ddr=dram_address, baddr_sram=self.bmem[bmem_not_in_use]
        )
        request_id = self.add_request_from_rdesc(mem_id, descriptor)
        if self.rdma_des_queue:
            self.rdma_des_queue[-1].last = False
            self.rdma_des_queue[-1].next_dsc = descriptor.id
        self.rdma_des_queue.append(descriptor)
        self.wait_request_info[mem_id] = request_id
    
    def load_lut(self, lut_name):
        lut_size = self.lut_raw.size
        dram_address = self.get_dram_address(lut_name, 0)
        descriptor = Descriptor3D(
            0, lut_size, lut_size, 1, lut_size, 1,
            intenb=True, last=True, baddr_ddr=dram_address, baddr_sram=self.lut_raw
        )
        request_id = self.add_request_from_rdesc(self.lid_offset, descriptor)
        if self.rdma_des_queue:
            self.rdma_des_queue[-1].last = False
            self.rdma_des_queue[-1].next_dsc = descriptor.id
        self.rdma_des_queue.append(descriptor)
        self.wait_request_info[self.lid_offset] = request_id

    def access_tmem(self):
        pass

    def _transfer_packet_to_dram(self):
        time_gap = 0
        if not self.wdma_des_queue:
            #raise RuntimeError('WDMA descriptor table is empty')
            return time_gap
        current_descriptor = self.wdma_des_queue[0]
        if not self.wdma_enabled:
            self.wdma_enabled = True
            self.write_dram_addr = current_descriptor.baddr_ddr + current_descriptor.aoff * self.data_size
            self.aiter = 0
            self.biter = 0
            self.citer = 0
            required_space = current_descriptor.acnt * current_descriptor.bcnt * current_descriptor.ccnt
            while self.write_available_space < required_space:
                time_gap = self.wait_for_end_request(self.tid_offset)
            self.add_request(mem_id=-1, request_type=3, request_size=0, buf=None, offset=0, dram_address=0)
            self.write_available_space -= required_space
        while True:
            transfer_size = min(self.config.MIDAP.PACKET_SIZE, current_descriptor.acnt - self.aiter)
            if (self.buffer_write_pivot - self.buffer_read_pivot) % self.tmem.size < transfer_size:
                break
            request_id = self.add_request(
                mem_id=self.tid_offset,
                request_type=1,
                request_size=transfer_size,
                buf=np.take(self.tmem, range(self.buffer_read_pivot, self.buffer_read_pivot + transfer_size), mode='wrap'),
                offset=0,
                dram_address=self.write_dram_addr
            )
            self.buffer_read_pivot = (self.buffer_read_pivot + transfer_size) % self.tmem.size
            self.aiter += transfer_size
            self.write_dram_addr += transfer_size * self.data_size
            if self.aiter >= current_descriptor.acnt:
                self.biter += 1
                self.aiter -= current_descriptor.acnt
                self.write_dram_addr += (current_descriptor.boff * (-1) ** int(current_descriptor.bsub) - current_descriptor.acnt) * self.data_size
            if self.biter >= current_descriptor.bcnt:
                self.citer += 1
                self.biter -= current_descriptor.bcnt
                self.write_dram_addr += (current_descriptor.coff * (-1) ** int(current_descriptor.csub) \
                    - current_descriptor.boff * (current_descriptor.bcnt if current_descriptor.bcsync else 1) * (-1) ** int(current_descriptor.bsub)) * self.data_size
            if self.citer >= current_descriptor.ccnt:
                if self.wait_request_info[self.tid_offset] == -1:
                    self.wait_request_info[self.tid_offset] = request_id
                self.wdma_wait_list.append((request_id, current_descriptor.acnt * current_descriptor.bcnt * current_descriptor.ccnt))
                # Load the next descriptor
                self.wdma_des_queue.popleft()
                if not self.wdma_des_queue:
                    self.wdma_enabled = False
                else:
                    current_descriptor = self.wdma_des_queue[0]
                    self.write_dram_addr = current_descriptor.baddr_ddr + current_descriptor.aoff * self.data_size
                    self.aiter = 0
                    self.biter = 0
                    self.citer = 0
                    required_space = current_descriptor.acnt * current_descriptor.bcnt * current_descriptor.ccnt
                    while self.write_available_space < required_space:
                        time_gap = self.wait_for_end_request(self.tid_offset)
                    self.write_available_space -= required_space
                    self.add_request(mem_id=-1, request_type=3, request_size=0, buf=None, offset=0, dram_address=0)
        return time_gap

    def write_tmem(self, address, data, valid): # 
        time_gap = 0
        data_pivot = address % self.system_width
        write_size = self.num_wmem * reduce(lambda acc, cur: acc + int(cur), valid, 0)
        if self.buffer_write_pivot + write_size > self.tmem.size:
            pivot = self.tmem.size - self.buffer_write_pivot
            self.tmem[self.buffer_write_pivot:] = data[data_pivot : data_pivot + pivot]
            self.tmem[:write_size - pivot] = data[data_pivot + pivot : data_pivot + write_size]
        else:
            self.tmem[self.buffer_write_pivot : self.buffer_write_pivot + write_size] = data[data_pivot : data_pivot + write_size]
        self.buffer_write_pivot = (self.buffer_write_pivot + write_size) % self.tmem.size
        if (self.buffer_write_pivot - self.buffer_read_pivot) % self.tmem.size >= self.config.MIDAP.PACKET_SIZE:
            time_gap += self._transfer_packet_to_dram()
        return time_gap 

    def tmem_to_dram(
            self,
            data_name,
            dram_pivot_address,
            transfer_unit_size,
            transfer_offset,
            num_transfers,
            tmem_pivot_address = 0,
            ):
        wait_time = self._transfer_packet_to_dram()
        self.manager.stats.wait_write_dram(wait_time)

    def switch_tmem(self):
        pass

    def write_dram(self, data_name, address, data, offset, size): 
        return 0

    def load_host(self, data_name, size):
        dram_address = self.get_dram_address(data_name, 0)
        descriptor = Descriptor3D(
            0, size, size, 1, size, 1,
            intenb=True, last=True, baddr_ddr=dram_address, baddr_sram=None
        )
        request_id = self.add_request_from_rdesc(self.hid_offset, descriptor)
        self.rdma_des_queue.append(descriptor)
        self.wait_request_info[self.hid_offset] = request_id

    def write_host(self, data_name, data):
        pass

    def sync_host(self):
        return 0

    def add_request(
            self,
            mem_id = -1,
            request_type = 0, # 0: Read, 1: Write, 2: RDMA setup, 3: WDMA setup
            request_size = 0, # R/W Size
            buf = None, # memory object (numpy array)
            offset = 0, # memory (buffer) R/W start offset
            dram_address = 0, # target dram address
            num_requests = 1,
            ):
        # if buf is None:
        #     self.logger.error("Memory(Buffer) object must be specified!")
        #     raise RuntimeError()
        if request_type in [2, 3]:
            request = RequestInfo(request_type, request_size, buf, offset, dram_address, num_requests)
            qe = QueueElement(mem_id, self.sim_time, request)
            if request_type == 3:
                self.wqueue.append(qe)
            else:
                self.lp_rqueue.append(qe)
            return request.id

        if num_requests > 1 and request_size > self.config.MIDAP.PACKET_SIZE:
            self.logger.error("for merged requests, request size should not be larger than packet size [rsize: {}, psize: {}]".format(request_size, self.config.MIDAP.PACKET_SIZE))
            raise RuntimeError()
        request_time = self.sim_time
        request_id = -1

        # Split the request into small packets and return the last packet id.
        while request_size > 0:
            size = self.config.MIDAP.PACKET_SIZE
            if request_size < self.config.MIDAP.PACKET_SIZE:
                size = request_size
            request = RequestInfo(request_type, size, buf, offset, dram_address, num_requests)
            qe = QueueElement(mem_id, request_time, request)

            request_size -= size
            dram_address += (size * self.element_size)
            offset += size

            if request_type == 1:
                self.wqueue.append(qe)
            else:
                self.lp_rqueue.append(qe)

            request_id = request.id

        return request_id

    def run_rdma(self, packet):
        request = packet.request
        assert request.type in [0, 2]
        if packet.request_time < self.rdma_timer:
            self.memory_stat[packet.mem_id].wait_append(packet.request_time, self.rdma_timer - packet.request_time)
        self.rdma_timer = max(packet.request_time, self.rdma_timer)
        start_time = self.rdma_timer
        if self.rdma_timer > self.rdma_next_reset:
            self.rdma_timer += self.refresh_delay
            self.rdma_next_reset = self.rdma_timer + self.refresh_period
        if request.type == 0:
            latency = self.get_transfer_latency(request.size, addr = request.dram_address, write = False)
        elif request.type == 2:
            latency = self.dma_setup_delay
        self.rdma_timer += latency
        self.memory_stat[packet.mem_id].busy_append(start_time, self.rdma_timer - start_time)

    def run_wdma(self, packet):
        request = packet.request
        assert request.type in [1, 3]
        self.wdma_timer = max(packet.request_time, self.wdma_timer)
        start_time = self.wdma_timer
        if self.wdma_timer > self.wdma_next_reset:
            self.wdma_timer += self.refresh_delay
            self.wdma_next_reset = self.wdma_timer + self.refresh_period
        if request.type == 1:
            latency = self.get_transfer_latency(request.size, addr = request.dram_address, write = True)
        elif request.type == 3:
            latency = self.dma_setup_delay
        self.wdma_timer += latency
        self.memory_stat[packet.mem_id].busy_append(start_time, self.wdma_timer - start_time)

    def update_queue(self, mem_id = -1, sync = False):
        self.rdma_des_queue.clear()
        current_time = self.sim_time
        # mem_id < 1: Update DMA - HSIM Timer until t <= current_time
        # When mem_id >= 0: Run DMA until the request of id = target_request_id is finished
        if mem_id == -1:
            rcheck = wcheck = lambda t, md: t <= current_time
            ret_val = lambda: max(self.wdma_timer, self.rdma_timer)
        elif sync:
            rcheck = wcheck = lambda t, md: len(self.wqueue) > 0 or len(self.hp_rqueue) > 0 or len(self.lp_rqueue) > 0
            ret_val = lambda: max(self.wdma_timer, self.rdma_timer)
        else:
            last_request = self.wait_request_info[mem_id]
            if mem_id == self.tid_offset:   # write
                wcheck = lambda t, md: self.wait_request_info[md] == last_request
                rcheck = lambda t, md: t <= current_time
                ret_val = lambda: self.wdma_timer
            else:                           # read
                rcheck = lambda t, md: self.wait_request_info[md] == last_request
                wcheck = lambda t, md: t <= current_time
                ret_val = lambda: self.rdma_timer
        while rcheck(self.rdma_timer, mem_id):
            # do sth // junk code // not only the priority of the request but timing must be considered together
            # 1) req_time < dma_time : highest priority
            # 2) wqueue > hp_rqueue > lp_rqueue
            if len(self.lp_rqueue) == 0:
                if mem_id < 0 or mem_id == self. tid_offset:
                    break
                raise RuntimeError("Cannot find the request for mem id {}, wait_request_info {}".format(mem_id, self.wait_request_info))
            packet = self.lp_rqueue.popleft()
            self.run_rdma(packet)
            mid, _, request = packet
            dt = 0
            for offset in self.dram_offset_idx:
                if request.dram_address >= offset:
                    dt = self.dram_offset_idx[offset]
            dram_address = (request.dram_address - self.dram_offset[dt]) // self.data_size
            if self.wait_request_info[mid] == request.id:
                self.logger.debug("REQUEST ID {} FOR MEM {} is finished".format(self.wait_request_info[mid],mid))
                self.wait_request_info[mid] = -1 # Processed
            if request.buf is None:
                pass
            elif request.num_requests == 1:
                request.buf[request.offset:request.offset+request.size] = \
                        self.dram_data[dt][dram_address:dram_address + request.size]
            else:
                shape = (request.num_requests, request.size // request.num_requests)
                request.buf[:, request.offset:request.offset + shape[-1]] = \
                        self.dram_data[dt][dram_address:dram_address + request.size].reshape(*shape)
        while wcheck(self.wdma_timer, mem_id):
            if len(self.wqueue) == 0:
                if mem_id != self.tid_offset:
                    break
                raise RuntimeError("Cannot find the request for mem id {}, wait_request_info {}".format(mem_id, self.wait_request_info))
            packet = self.wqueue.popleft()
            self.run_wdma(packet)
            mid, _, request = packet
                # Write the request.buf on the DRAM
            dt = 0
            for offset in self.dram_offset_idx:
                if request.dram_address >= offset:
                    dt = self.dram_offset_idx[offset]
            dram_address = (request.dram_address - self.dram_offset[dt]) // self.data_size
            if self.wait_request_info[mid] == request.id:
                self.logger.debug("REQUEST ID {} FOR MEM {} is finished".format(self.wait_request_info[mid],mid))
                self.wait_request_info[mid] = -1 # Processed
                if not self.wdma_wait_list:
                    raise RuntimeError('wdma_wait_list is empty')
                self.write_available_space += self.wdma_wait_list.popleft()[1]
                if self.wdma_wait_list:
                    self.wait_request_info[mid] = self.wdma_wait_list[0][0]
            # Process packet
            if request.buf is not None:
                self.logger.debug("Process REQUEST {} , From TMEM {} to DRAM Address {}.. ".format(request,mid,dram_address))
                self.dram_data[dt][dram_address: dram_address + request.size] = request.buf[request.offset:request.offset + request.size]
        # return value
        return math.ceil(ret_val()) # DMA Time

    def wait_sync(self, wait_id):
        self.rdma_next_reset = 0
        self.wdma_next_reset = 0
        return super().wait_sync(wait_id)
