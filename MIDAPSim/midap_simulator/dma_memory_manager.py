from __future__ import absolute_import, division, print_function, unicode_literals

import math
import numpy as np
from collections import deque

from config import cfg
from .memory_manager import MemoryManager
from .packet_manager import PacketManager

class RequestInfo(object): # Memorymanager ---(QUEUE)---> DMA
    request_id = 0
    def __init__(
            self,
            request_type = 0, # 0: Read, 1: Write
            request_size = 0, # R/W Size
            buf = None, # memory object (numpy array)
            offset = 0, # memory (buffer) R/W start offset
            dram_address = 0, # target dram address
            num_requests = 1,
            ):
        self.id = RequestInfo.request_id
        RequestInfo.request_id += 1
        self.type = request_type
        self.size = request_size
        # In fact, on-chip info should be specified as 'Integrated Address'
        # But, for simple simulation, on-chip data info is specified with [Memory Object & Offset] Format
        # Otherwise, memory object can be replaced with memory id
        self.buf = buf # Memory object
        self.offset = offset
        # DRAM Info
        self.dram_address = dram_address
        self.num_requests = num_requests

    def __repr__(self):
        ret = "(ID, Size, Offset) = ({}, {}, {})\n".format(self.id, self.size, self.offset)
        return ret

class QueueElement(list):
    def __init__(self, mem_id, request_time, request, layer_name=""):
        super().__init__([mem_id, request_time, request])
        self.mem_id = mem_id
        self.request_time = request_time
        self.request = request
        self.layer_name = layer_name

class DMemoryManager(MemoryManager):
    # DMA-based manage 
    # Interface: Donghyun
    # Implementation: Keonjoo
    def __init__(self, manager):
        super().__init__(manager)
        self.dram_offset_idx = dict(sorted({self.dram_offset[i]: i for i in range(len(self.dram_offset))}.items()))
        self.lp_rqueue = deque() # Low priority request queue
        self.hp_rqueue = deque() # High priority request queue
        self.wqueue = deque() # Write queue
        self.wait_request_info = [-1 for _ in range(2 + 2 + 2 + 1 + self.num_fmem + 1)] # add host
        self.id_offsets = [0, 2, 4, 6, 7, 7 + self.num_fmem]
        self.queue_crits = [self.id_offsets[1], self.id_offsets[2]]
        self.memory_stat = self.tmem_stat + self.wmem_stat + self.bmem_stat + [self.lut_stat] + self.fmem_stat + [self.host_stat]
        self.time = 0 #DMA Time
        self.last_time = 0

        self.element_size = 4 if not self.config.MODEL.QUANTIZED else 1

        # Communicate with HSIM
        if self.config.MIDAP.CORE_ID >= 0 and self.config.DRAM.COMM_TYPE == 'DMA':
            self.packet_manager = PacketManager(".args.shmem.dat_" + str(self.config.MIDAP.CORE_ID), self.config.MODEL.QUANTIZED)

    def __del__(self):
        if self.config.MIDAP.CORE_ID >= 0 and self.config.DRAM.COMM_TYPE == 'DMA':
            self.packet_manager.terminatedRequest(self.sim_time)
            del self.packet_manager
    
    @property
    def tid_offset(self):
        return self.id_offsets[0]

    @property
    def wid_offset(self):
        return self.id_offsets[1]
    
    @property
    def bid_offset(self):
        return self.id_offsets[2]

    @property
    def lid_offset(self):
        return self.id_offsets[3]

    @property
    def fid_offset(self):
        return self.id_offsets[4]
    
    @property
    def hid_offset(self):
        return self.id_offsets[5]

    @property
    def wqueue_crit(self):
        return self.queue_crits[0]
    
    @property
    def hp_queue_crit(self):
        return self.queue_crits[1]
    
    @property
    def dram_address_dict(self):
        return self.dram_dict

    def set_dram_info(self, dram_data, dram_dict):
        self.dram_dict = dram_dict

    def reset_wmem(self):
        self.wmem_in_use = -1
        for i in range(2):
            self.logger.info("reset_wmem idx:{}, request_id:{}".format(self.wid_offset + i, self.wait_request_info[self.wid_offset + i]))
            if self.wait_request_info[self.wid_offset + i] != -1:
                self.logger.error("WMEM Request information must be empty before reset. WMEM {}: {} {}".format(i, self.wait_request_info[self.wid_offset + i], self.wid_offset + i))
                # Or, it must be a wrong reset timing 
                raise RuntimeError()
    
    ########################### TODO: If the data request size is too huge, you may split the request into small packets (ex: 2KiB)
    def add_request(
            self,
            mem_id = -1,
            request_type = 0, # 0: Read, 1: Write
            request_size = 0, # R/W Size
            buf = None, # memory object (numpy array)
            offset = 0, # memory (buffer) R/W start offset
            dram_address = 0, # target dram address
            num_requests = 1,
            layer_name = ""
            ):
        # if buf is None:
        #     self.logger.error("Memory(Buffer) object must be specified!")
        #     raise RuntimeError()
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
            qe = QueueElement(mem_id, request_time, request, layer_name)

            request_size -= size
            dram_address += (size * self.element_size)
            offset +=  size

            if request_type == 1:
                self.wqueue.append(qe)        
            elif mem_id < self.hp_queue_crit: 
                self.hp_rqueue.append(qe)           
            else:
                self.lp_rqueue.append(qe)         
            request_id = request.id

        return request_id
    ###########################

    def get_dram_address(self, name, offset):
        dt, address = self.dram_address_dict[name]
        return self.dram_offset[dt] + (address + offset) * self.data_size

    def load_wmem(
            self,
            to_all,
            filter_name,
            filter_size = 0,
            filter_offset = 0,
            wmem_offset = 0,
            dram_offset = 0,
            continuous_request = False,
            layer_name = ""):
        # TODO: determine the data written checking (when the wmem data is the result of previous layers)
        wmem_not_in_use = (self.wmem_in_use + 1) % 2
        mem_id = self.wid_offset + wmem_not_in_use
        self.logger.info("{}::Load wmem [{}] size {} to WMEM".format(layer_name, filter_name, dram_offset, filter_size))
        for wmem_idx in range(self.num_wmem if to_all else 1):
            dram_address = self.get_dram_address(filter_name, dram_offset + wmem_idx * filter_offset)
            self.update_queue()
            request_id = self.add_request(
                    mem_id,
                    0,
                    filter_size,
                    self.wmem[wmem_not_in_use][wmem_idx],
                    wmem_offset,
                    dram_address,
                    1,
                    layer_name)
            self.continuous_request_info = [-1, 0, 0, 0]
            self.wait_request_info[self.wid_offset + wmem_not_in_use] = request_id
    
    def load_fmem(self, fmem_idx, data_name, data_size, fmem_offset = 0, dram_offset = 0, layer_name = ""):
        # TODO: determine the data written checking (when the fmem data is the result of previous layers)
        self.update_queue() # Not necessary
        dram_address = self.get_dram_address(data_name, dram_offset)
        self.logger.info("{}::Load fmem [{}] size:{}, address:{}".format(layer_name, data_name, data_size, dram_address))
        mem_id = self.fid_offset + fmem_idx
        request_id = self.add_request(
                mem_id,
                0,
                data_size,
                self.fmem[fmem_idx],
                fmem_offset,
                dram_address,
                1,
                layer_name
                )
        if self.wait_request_info[mem_id] != -1:
            self.logger.error("FMEM Request information must be empty before the new request. FMEM {}: {}".format(fmem_idx, self.wait_request_info[mem_id]))
        self.wait_request_info[mem_id] = request_id

    def load_bmem(self, bias_name, bias_size):
        self.update_queue() # Not necessary 
        bmem_not_in_use = (self.bmem_in_use + 1) % 2
        dram_address = self.get_dram_address(bias_name, 0)
        mem_id = self.bid_offset + bmem_not_in_use
        request_id = self.add_request(
                mem_id,
                0,
                bias_size,
                self.bmem[bmem_not_in_use],
                0,
                dram_address)
        if self.wait_request_info[mem_id] != -1:
            self.logger.error("BMEM Request information must be empty before the new request. BMEM {}: {}".format(bmem_not_in_use, self.wait_request_info[mem_id]))
        self.wait_request_info[mem_id] = request_id
        self.logger.info("Time {}: LOAD BMEM bank {} (Data - {}), size {}".format(self.sim_time, bmem_not_in_use, bias_name, bias_size)) 
    
    def load_lut(self, lut_name):
        lut_size = self.lut_raw.size
        self.update_queue() # Not necessary 
        dram_address = self.get_dram_address(lut_name, 0)
        mem_id = self.lid_offset
        request_id = self.add_request(
                mem_id,
                0,
                lut_size,
                self.lut_raw,
                0,
                dram_address)
        if self.wait_request_info[mem_id] != -1:
            self.logger.error("BMEM Request information must be empty before the new request. LUT {}".format(self.wait_request_info[mem_id]))
        self.wait_request_info[mem_id] = request_id
        self.logger.info("Time {}: LOAD LUT (Data - {}), size {}".format(self.sim_time, lut_name, lut_size)) 
    
    #New Interface for write.. with buffer
    def tmem_to_dram( # Blocking Write..
            self,
            data_name,
            dram_pivot_offset,
            transfer_unit_size,
            transfer_offset,
            num_transfers,
            tmem_pivot_address = 0,
            ):
        # Update timer
        dram_pivot_address = self.get_dram_address(data_name, dram_pivot_offset)
        mem_id = self.tid_offset + self.tmem_in_use
        requests = [] # Debug
        for idx in range(num_transfers):
            dram_address = dram_pivot_address + self.data_size * transfer_offset * idx
            tmem_address = tmem_pivot_address + transfer_unit_size * idx
            request_id = self.add_request(
                    mem_id,
                    1,
                    transfer_unit_size,
                    self.tmem[self.tmem_in_use],
                    tmem_address,
                    dram_address)
            requests.append(request_id)
        if self.wait_request_info[mem_id] != -1:
            self.logger.error("TMEM Request information must be empty before the new request. TMEM {}: {}".format(self.tmem_in_use, self.wait_request_info[mem_id]))
        self.wait_request_info[mem_id] = request_id
        self.logger.debug("TMEM {} is waiting for requests [{},{}]".format(self.tmem_in_use, requests[0], requests[-1]))

    def load_host(self, data_name, size):
        self.update_queue() # Not necessary 
        dram_address = self.get_dram_address(data_name, 0)
        mem_id = self.hid_offset
        request_id = self.add_request(
                mem_id,
                0,
                size,
                None,
                0,
                dram_address)
        if self.wait_request_info[mem_id] != -1:
            self.logger.error("Host Request information must be empty before the new request. LUT {}".format(self.wait_request_info[mem_id]))
        self.wait_request_info[mem_id] = request_id
        self.logger.info("Time {}: HOST LOAD (Data - {}), size {}".format(self.sim_time, data_name, size)) 
    
    #New Interface for write.. with buffer
    def write_host( # Blocking Write..
            self,
            data_name,
            data
            ):
        # Update timer
        self.update_queue() # Not necessary 
        dram_address = self.get_dram_address(data_name, 0)
        mem_id = self.hid_offset
        requests = [] # Debug
        request_id = self.add_request(
                mem_id,
                1,
                data.size,
                data,
                0,
                dram_address
                )
        requests.append(request_id)
        if self.wait_request_info[mem_id] != -1:
            self.logger.error("HOST WRITE information must be empty before the new request. : {}".format(self.wait_request_info[mem_id]))
        self.wait_request_info[mem_id] = request_id
        self.logger.info("Time {}: HOST WRITE (Data - {}), size {}".format(self.sim_time, data_name, data.size)) 

    def sync_host(self):
        current_time = self.sim_time
        time_gap = 0
        mem_id = self.hid_offset
        if self.wait_request_info[mem_id] >= 0:
            time_gap = self.wait_for_end_request(mem_id)
            if time_gap > 0:
                self.logger.debug("HOST delay occured & time_gap = {}".format(time_gap))
                self.memory_stat[mem_id].delay_append(current_time, time_gap)
        return math.ceil(time_gap)

    def access_wmem(self):
        # Wait for the Request
        current_time = self.sim_time
        time_gap = 0
        mem_id = self.wid_offset + self.wmem_in_use
        if self.wait_request_info[mem_id] >= 0:
            time_gap = self.wait_for_end_request(mem_id)
            if time_gap > 0:
                self.logger.debug("WMEM {} load delay occured & time_gap = {}".format(mem_id, time_gap))
                self.memory_stat[mem_id].delay_append(current_time, time_gap)
        return math.ceil(time_gap)

    def access_fmem(self, bank_idx):
        current_time = self.sim_time
        time_gap = 0
        mem_id = self.fid_offset + bank_idx
        if self.wait_request_info[mem_id] >= 0:
            time_gap = self.wait_for_end_request(mem_id)
            if time_gap > 0:
                self.logger.debug("FMEM {} load delay occured & time_gap = {}".format(bank_idx, time_gap))
                self.memory_stat[mem_id].delay_append(current_time, time_gap)
        return math.ceil(time_gap)

    def access_bmem(self):
        current_time = self.sim_time
        time_gap = 0
        mem_id = self.bid_offset + self.bmem_in_use
        if self.wait_request_info[mem_id] >= 0:
            time_gap = self.wait_for_end_request(mem_id)
            if time_gap > 0:
                self.logger.debug("BMEM {} load delay occured & time_gap = {}".format(self.bmem_in_use, time_gap))
                self.memory_stat[mem_id].delay_append(current_time, time_gap)
        return math.ceil(time_gap)
    
    def access_lut(self):
        current_time = self.sim_time
        time_gap = 0
        mem_id = self.lid_offset
        if self.wait_request_info[mem_id] >= 0:
            time_gap = self.wait_for_end_request(mem_id)
            if time_gap > 0:
                self.logger.debug("LUT load delay occured & time_gap = {}".format(time_gap))
                self.memory_stat[mem_id].delay_append(current_time, time_gap)
        return math.ceil(time_gap)

    def access_tmem(self):
        current_time = self.sim_time
        time_gap = 0
        mem_id = self.tid_offset + self.tmem_in_use
        if self.wait_request_info[mem_id] >= 0:
            time_gap = self.wait_for_end_request(mem_id)
            if time_gap > 0:
                self.logger.debug("TMEM {} access delay (DRAM Write delay) occured & time_gap = {}".format(self.tmem_in_use, time_gap))
                self.memory_stat[mem_id].delay_append(current_time, time_gap)
        return math.ceil(time_gap)

    def elapse_cycle(self):
        if self.sim_time < self.time:
            return

        min_time = self.sim_time
        if len(self.wqueue) > 0:
            _, req_time, _ = self.wqueue[0]
            if req_time < min_time:
                min_time = req_time
        if len(self.hp_rqueue) > 0:
            _, req_time, _ = self.hp_rqueue[0]
            if req_time < min_time:
                min_time = req_time
        if len(self.lp_rqueue) > 0:
            _, req_time, _ = self.lp_rqueue[0]
            if req_time < min_time:
                min_time = req_time

        if self.config.MIDAP.CORE_ID >= 0:
            self.packet_manager.elapsedRequest(min_time)

        if min_time > self.time:
            self.time = min_time

    ###### TODO: DMA-aware Implementation
    def update_queue(self, mem_id = -1, sync = False):
        reqToRead = None

        current_time = self.sim_time
        # mem_id < 1: Update DMA - HSIM Timer until t <= current_time
        # When mem_id >= 0: Run DMA until the request of id = target_request_id is finished
        # time = 0 #DMA Time
        if mem_id == -1:
            check = lambda t, md: t <= current_time
        elif sync:
            check = lambda t, md: len(self.wqueue) > 0 or len(self.hp_rqueue) > 0 or len(self.lp_rqueue) > 0
        else:
            check = lambda t, md: self.wait_request_info[md] >= 0
        while check(self.time, mem_id) or reqToRead is not None:
            if len(self.wqueue) == 0 and len(self.hp_rqueue) == 0 and len(self.lp_rqueue) == 0 and reqToRead is None:
                self.time = current_time
                self.packet_manager.elapsedRequest(current_time)
                break

            if len(self.wqueue) > 0:
                w_mid, req_time, request = self.wqueue.popleft()
                # self.logger.info("[MIDAP] writeRequest:{}:{}".format(request.size,request.dram_address))
                self.packet_manager.writeRequest(request.dram_address, request.size, request.buf[request.offset:request.offset+request.size], self.time)
                self.time = max(self.time, req_time)
                if self.wait_request_info[w_mid] == request.id:
                    self.wait_request_info[w_mid] = -1 # Processed

            if reqToRead is not None:
                data, work_time, address = self.packet_manager.readResponse(reqToRead.size)
                if data is not None:
                    # self.logger.info("[MIDAP] readResponse:{}:{}, {}".format(reqToRead.size, address, self.time))
                    if reqToRead.buf is not None:
                        if reqToRead.num_requests == 1:
                            data = data.flatten()
                            reqToRead.buf[reqToRead.offset:reqToRead.offset+reqToRead.size] = data[:reqToRead.size]
                        else:
                            shape = (reqToRead.num_requests, reqToRead.size // reqToRead.num_requests)
                            reqToRead.buf[:, reqToRead.offset:reqToRead.offset + shape[-1]] = data[:reqToRead.size].reshape(*shape)
                    if work_time > 0 and self.time < work_time:
                        self.time = work_time
                    if self.wait_request_info[mid] == reqToRead.id:
                        self.wait_request_info[mid] = -1 # Processed

                    reqToRead = None

            if reqToRead is None:
                high_prior = False
                request = None
                layer_name = None
                if len(self.hp_rqueue) > 0:
                    # mid, req_time, request = self.hp_rqueue.popleft()
                    qe = self.hp_rqueue.popleft()
                    mid = qe.mem_id
                    req_time = qe.request_time
                    request = qe.request
                    layer_name = qe.layer_name
                    high_prior = True
                elif len(self.lp_rqueue) > 0:
                    mid, req_time, request = self.lp_rqueue.popleft()

                if request is not None:
                    # if layer_name is not "null":
                        # self.logger.info("[MIDAP] readRequest:{}:{}".format(layer_name, request.dram_address))
                    self.packet_manager.readRequest(request.dram_address, request.size, self.time, high_prior)
                    reqToRead = request
        return math.ceil(self.time) # DMA Time
    ###### TODO End


    def wait_for_end_request(self, mem_id):
        current_time = self.sim_time
        end_time = self.update_queue(mem_id = mem_id)
        if end_time < current_time:
            self.update_queue()
        return max(0, end_time - current_time)

    def sync(self, sync_id, compute_cycle, layer_id):
        self.update_queue(mem_id = 0, sync = True)
        self.packet_manager.signalRequest(sync_id, self.time, compute_cycle, layer_id)

    def wait_sync(self, wait_id, layer_id):
        sync_time = self.packet_manager.waitRequest(wait_id, self.time, self.sim_time, layer_id)
        if sync_time > 0 and sync_time > self.time:
            self.time = math.ceil(sync_time)
        time_gap = max(0, math.ceil(self.time) - self.sim_time)
        return time_gap

class TDMemoryManager(DMemoryManager): # Test dump memory file & DMA logic works
    def __init__(self, manager):
        super().__init__(manager)
        self.dma_timer = 0
        self.next_reset = 0
        self.dram_constants = [self.config.DRAM.CAS, self.config.DRAM.PAGE_DELAY, self.config.DRAM.REFRESH_DELAY]
        self.dram_page_size = self.config.DRAM.PAGE_SIZE
        self.refresh_period = self.config.DRAM.REFRESH_PERIOD
        self.dram_bandwidth = (self.config.SYSTEM.BANDWIDTH * 1000 / self.config.SYSTEM.FREQUENCY) / self.config.SYSTEM.DATA_SIZE
        self.bus_bandwidth = self.config.DRAM.BUS_BANDWIDTH
        self.last_read_address = -1

    @property
    def cas(self):
        return self.dram_constants[0]

    @property
    def page_delay(self):
        return self.dram_constants[1]

    @property
    def refresh_delay(self):
        return self.dram_constants[2]

    def set_dram_info(self, dram_data, dram_dict):
        self.dram_dict = dram_dict
        self.dram_data = dram_data
    
    def get_dram_read_latency(self, size, addr):
        cas, pg_dly, rst_dly = self.dram_constants
        assert size <= self.config.MIDAP.PACKET_SIZE
        access_new_pg = 1 if (addr % self.dram_page_size) < self.config.MIDAP.PACKET_SIZE else 0
        predict = pg_dly * access_new_pg + cas
        return predict

    def get_transfer_latency(self, size, addr = None, write = False):
        t = math.ceil(size / self.dram_bandwidth) # DRAM Transfer Time
        if not write:
            t += self.get_dram_read_latency(size, addr) ## DMA: blocking R&W 
        else:
            t += self.page_delay * size / self.dram_page_size
        t += math.ceil(size / self.bus_bandwidth) # BUS Delay
        t += 4  # Bus Overhead per each packet = 4 cycles
        return t

    def run_bus(self, packet):
        request = packet.request
        if packet.request_time < self.dma_timer:
            self.memory_stat[packet.mem_id].wait_append(packet.request_time, self.dma_timer - packet.request_time)
        self.dma_timer = max(packet.request_time, self.dma_timer)
        start_time = self.dma_timer
        if self.dma_timer > self.next_reset:
            self.dma_timer += self.refresh_delay
            self.next_reset = self.dma_timer + self.refresh_period
        latency = self.get_transfer_latency(request.size, addr = request.dram_address, write = request.type == 1)
        self.dma_timer += latency
        self.memory_stat[packet.mem_id].busy_append(start_time, self.dma_timer - start_time)

    def update_queue(self, mem_id = -1, sync = False):
        current_time = self.sim_time
        # mem_id < 1: Update DMA - HSIM Timer until t <= current_time
        # When mem_id >= 0: Run DMA until the request of id = target_request_id is finished
        if mem_id == -1:
            check = lambda t, md: t <= current_time
        elif sync:
            check = lambda t, md: len(self.wqueue) > 0 or len(self.hp_rqueue) > 0 or len(self.lp_rqueue) > 0
        else:
            check = lambda t, md: self.wait_request_info[md] >= 0
        while check(self.dma_timer, mem_id):
            # do sth // junk code // not only the priority of the request but timing must be considered together
            # 1) req_time < dma_time : highest priority
            # 2) wqueue > hp_rqueue > lp_rqueue
            if len(self.wqueue) > 0:
                packet = self.wqueue.popleft()
            elif len(self.hp_rqueue) > 0:
                packet = self.hp_rqueue.popleft()
            else:
                if len(self.lp_rqueue) == 0:
                    if mem_id < 0:
                        break
                    raise RuntimeError("Cannot find the request for mem id {}, wait_request_info {}".format(mem_id, self.wait_request_info))
                packet = self.lp_rqueue.popleft()
            self.run_bus(packet)
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
            # Process packet
            if request.buf is None:
                pass
            elif request.type == 1:
                self.logger.debug("Process REQUEST {} , From TMEM {} to DRAM Address {}.. ".format(request,mid,dram_address))
                self.dram_data[dt][dram_address: dram_address + request.size] = request.buf[request.offset:request.offset + request.size]
            elif request.num_requests == 1:
                request.buf[request.offset:request.offset+request.size] = \
                        self.dram_data[dt][dram_address:dram_address + request.size]
            else:
                shape = (request.num_requests, request.size // request.num_requests)
                request.buf[:, request.offset:request.offset + shape[-1]] = \
                        self.dram_data[dt][dram_address:dram_address + request.size].reshape(*shape)
        return math.ceil(self.dma_timer) # DMA Time

    def sync(self, sync_id):
        from .shared_info import SyncManager
        self.update_queue(mem_id = 0, sync = True)
        SyncManager.sync(sync_id)
        self.logger.debug(f"Sync: Sync list = {SyncManager.get_sync_layers()}")

    def wait_sync(self, wait_id):
        from .shared_info import SyncManager, PrefetchManager
        check = SyncManager.check_sync(wait_id)
        if check:
            self.logger.debug(f"Sync wait success: {wait_id}")
        else:
            if wait_id[0] == 0:
                while not check:
                    proc_cnt = PrefetchManager.run_unit_transfer_single_core(self.core_id, self.dram_data)
                    if proc_cnt == 0:
                        self.logger.error(f"Sync wait failed!! {wait_id} not in {SyncManager.get_sync_layers()}: Deadlock or critical error")
                        raise RuntimeError
                    check = SyncManager.check_sync(wait_id)
            elif wait_id[0] == self.core_id:
                self.logger.error(f"Sync wait failed!! {wait_id} not in {SyncManager.get_sync_layers()}: Critical error")
                raise RuntimeError
            else:
                self.logger.info(f"Sync wait failed!! {wait_id} not in {SyncManager.get_sync_layers()}: Critical Error")
                raise RuntimeError
        self.next_refresh = 0 # Refresh must occur at the first dram request of each layer
        return 0
