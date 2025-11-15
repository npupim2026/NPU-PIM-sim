from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import math
import copy

from .memory_manager import MemoryManager, MemoryStat

from .shared_info import PrefetchManager, SyncManager


class VMemoryManager(MemoryManager): # Deprecated... please use TVMemoryManager instead of VMemoryManager.
    def __init__(self, manager):
        super().__init__(manager)
        # Set DRAM constraints
        self.config = manager.config
        self.bus_policy = self.config.MIDAP.BUS_POLICY
        self.dram_constants = [self.config.DRAM.CAS, self.config.DRAM.PAGE_DELAY, self.config.DRAM.REFRESH_DELAY]
        self.dram_page_size = self.config.DRAM.PAGE_SIZE
        self.dram_offsets = [self.config.DRAM.REFRESH_PERIOD]
        self.dram_bandwidth = (self.config.SYSTEM.BANDWIDTH * 1000 // self.config.SYSTEM.FREQUENCY) // self.config.SYSTEM.DATA_SIZE
        self.bus_bandwidth = self.config.DRAM.BUS_BANDWIDTH
        self.dram_latency_type = self.config.LATENCY.LATENCY_TYPE.lower()
        self.include_dram_write = self.config.DRAM.INCLUDE_DRAM_WRITE
        self.fmem_valid_timer = [-1 for _ in range(self.num_fmem)]
        self.wmem_valid_timer = [-1, -1]
        self.tmem_valid_timer = [-1, -1]
        self.bmem_valid_timer = [0, 0]
        self.lut_valid_timer = 0
        self.host_valid_timer = 0
        self.write_timer = 0
        self.dma_timer = 0
        self.continuous_request_size = 0
        self.next_refresh = 0
        self.refresh_delay = self.dram_constants[-1]
        self.refresh_period = int(self.dram_offsets[0]) # 4 * self.config.SYSTEM.FREQUENCY

    def store(self):
        return [self.wmem_in_use,
                self.bmem_in_use,
                self.next_refresh,
                self.dma_timer,
                self.write_timer,
                copy.deepcopy(self.fmem_valid_timer),
                copy.deepcopy(self.wmem_valid_timer),
                copy.deepcopy(self.bmem_valid_timer),
                self.lut_valid_timer,
                copy.deepcopy(self.fmem_stat),
                copy.deepcopy(self.wmem_stat),
                copy.deepcopy(self.tmem_stat),
                copy.deepcopy(self.bmem_stat),
                copy.deepcopy(self.lut_stat),
                copy.deepcopy(self.host_stat),
                copy.deepcopy(self.write_stat)]
    
    def restore(self, save):
        self.wmem_in_use, self.bmem_in_use, self.next_refresh, self.dma_timer, self.write_timer, fmem_valid_timer,\
        wmem_valid_timer, bmem_valid_timer, self.lut_valid_timer, fmem_stat, wmem_stat, tmem_stat, bmem_stat,\
        lut_stat, host_stat, write_stat = save
        self.fmem_valid_timer = copy.deepcopy(fmem_valid_timer)
        self.wmem_valid_timer = copy.deepcopy(wmem_valid_timer)
        self.bmem_valid_timer = copy.deepcopy(bmem_valid_timer)
        self.fmem_stat = copy.deepcopy(fmem_stat)
        self.wmem_stat = copy.deepcopy(wmem_stat)
        self.tmem_stat = copy.deepcopy(tmem_stat)
        self.bmem_stat = copy.deepcopy(bmem_stat)
        self.lut_stat = copy.deepcopy(lut_stat)
        self.host_stat = copy.deepcopy(host_stat)
        self.write_stat = copy.deepcopy(write_stat)

    def get_dram_read_latency(self, size):
        if self.dram_latency_type == 'worst':
            cas, pg_dly, rst_dly = self.dram_constants
            rst_prd = self.refresh_period
            predict = cas * math.ceil(size / self.config.MIDAP.PACKET_SIZE) + pg_dly * (size / self.dram_page_size)
            return predict
        elif self.dram_latency_type == 'exact':
            cas, pg_dly, rst_dly = self.dram_constants
            predict = pg_dly * math.ceil(size / self.dram_page_size) + cas * math.ceil(size / self.config.MIDAP.PACKET_SIZE)
            return predict
        else:
            raise ValueError("Unknown latency type!: " + self.dram_latency_type)
    
    def get_transfer_latency(self, size):
        t = math.ceil(size / self.dram_bandwidth) # DRAM Transfer Time
        t += self.get_dram_read_latency(size) ## DMA: blocking R&W 
        # Bus Overhead = 4 cycles * number of packet + data size divided by bus width (128)
        t += 4 * math.ceil(size / self.config.MIDAP.PACKET_SIZE) + math.ceil(size / (self.system_width * 2))
        if self.dram_latency_type == 'exact':
            dma_timer = max(self.dma_timer, self.sim_time)
            rf_offset = self.refresh_period
            if self.next_refresh + self.refresh_delay < dma_timer:
                rf_offset = 0
            elif dma_timer < self.next_refresh:
                rf_offset -= min(self.refresh_period, self.next_refresh - dma_timer)
            ref_time = int(rf_offset + t)
            ref_cnt = ref_time // self.refresh_period
            update = ref_cnt > 0
            #print("rf_offset, ref_time, ref_cnt: {}".format((rf_offset, ref_time, ref_cnt)))
            while ref_cnt > 0:
                ref_dly = self.refresh_delay * ref_cnt
                t += ref_dly
                ref_time = (ref_time % self.refresh_period) + ref_dly
                ref_cnt = ref_time // self.refresh_period
            if update:
                self.next_refresh = dma_timer + math.ceil(t / self.refresh_period) * self.refresh_period
        else:
            t += int(self.refresh_delay * t / self.refresh_period)
        # t += math.ceil(size / self.system_width) #
        return t
    
    def sync(self, sync_id):
        SyncManager.sync(sync_id)
        self.logger.debug(f"Sync: Sync list = {SyncManager.get_sync_layers()}")

    def wait_sync(self, wait_id):
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
        if self.write_timer > self.sim_time:
            self.write_stat.delay_append(self.sim_time, self.write_timer - self.sim_time)
        return max(0, self.write_timer - self.sim_time)
        
    def reset_wmem(self):
        self.wmem_in_use = -1
    
    def read_dram_data(self, name, offset, size):
        return self.dram_dict[name][offset:offset + size]

    def load_wmem(self, to_all, filter_name, filter_size = 0, filter_offset = 0, wmem_offset = 0, dram_offset = 0, continuous_request = False):
        wmem_not_in_use = (self.wmem_in_use + 1) % 2
        #self.logger.debug("Load data [{}] - addr {}, size {} to WMEM {}, offset {}".format(filter_name, dram_offset, filter_size, (wmem_not_in_use, wmem_idx), wmem_offset))
        if to_all and wmem_offset + filter_size > self.wmem_size:
            self.logger.error("WMEM Size: {} vs Requested Address: {}".format(self.wmem_size, wmem_offset + filter_size))
            raise ValueError("Wrong Address")
        for wmem_idx in range(self.num_wmem if to_all else 1):
            self.wmem[wmem_not_in_use, wmem_idx, wmem_offset:wmem_offset + filter_size] = \
                    self.read_dram_data(filter_name, dram_offset + wmem_idx * filter_offset, filter_size)
            # DRAM Access time
            if False and continuous_request:
                self.continuous_request_size += filter_size
                self.logger.debug("Transfer size cumulation...")
                return None
            # Update WMEM Timer Info
            load_start_time = max(self.sim_time, *self.tmem_valid_timer, *self.wmem_valid_timer)
            expected_transfer_time = self.get_transfer_latency(filter_size)
            wait_start_time = max(self.sim_time, self.wmem_valid_timer[wmem_not_in_use])
            if load_start_time > wait_start_time:
                self.wmem_stat[wmem_not_in_use].wait_append(wait_start_time, load_start_time - wait_start_time)
            self.wmem_valid_timer[wmem_not_in_use] = load_start_time + expected_transfer_time
            self.wmem_stat[wmem_not_in_use].busy_append(load_start_time, expected_transfer_time)
            required_update_fn = [self.update_dma_timer, self.update_bmem_timer, self.update_fmem_timer]
            for update_timer in required_update_fn:
                update_timer(load_start_time, expected_transfer_time)
            if not continuous_request:
                self.logger.debug("WMEM {} timer is updated to {}, dma_timer = {}".format(
                    wmem_not_in_use, self.wmem_valid_timer[wmem_not_in_use], self.dma_timer))
            self.continuous_request_size = 0
    
    def load_fmem(self, fmem_idx, data_name, data_size, fmem_offset = 0, dram_offset = 0):
        self.logger.debug("Load data [{}] - addr {}, size {} to FMEM {}".format(data_name, dram_offset, data_size, fmem_idx))
        self.fmem[fmem_idx, fmem_offset:fmem_offset+data_size] = \
                self.read_dram_data(data_name, dram_offset, data_size)
        # DRAM Access Time
        latency = self.get_transfer_latency(data_size)
        wait_start_time = max(self.sim_time, self.fmem_valid_timer[fmem_idx])
        if self.dma_timer > wait_start_time:
            self.fmem_stat[fmem_idx].wait_append(wait_start_time, self.dma_timer - wait_start_time)
        self.update_dma_timer(self.sim_time, latency)
        self.fmem_valid_timer[fmem_idx] = self.dma_timer
        self.fmem_stat[fmem_idx].busy_append(self.dma_timer - latency, latency)
        self.logger.debug("FMEM {} timer & dma_timer is updated to {}".format(fmem_idx, self.dma_timer))
    
    def load_bmem(self, bias_name, bias_size):
        bmem_not_in_use = (self.bmem_in_use + 1) % 2
        self.bmem[bmem_not_in_use, : bias_size] = self.read_dram_data(bias_name, 0, bias_size)
        #DRAM Access Time
        latency = self.get_transfer_latency(bias_size)
        wait_start_time = max(self.sim_time, self.bmem_valid_timer[bmem_not_in_use])
        if self.dma_timer > wait_start_time:
            self.bmem_stat[bmem_not_in_use].wait_append(wait_start_time, self.dma_timer - wait_start_time)
        self.update_dma_timer(self.sim_time, latency)
        self.bmem_valid_timer[bmem_not_in_use] = self.dma_timer
        self.bmem_stat[bmem_not_in_use].busy_append(self.dma_timer - latency, latency)
        self.logger.debug("BMEM {} timer & dma_timer is updated to {}".format(bmem_not_in_use, self.dma_timer))

    def load_lut(self, lut_name):
        lut_size = self.lut.size
        self.lut_raw[:] = self.read_dram_data(lut_name, 0, lut_size).astype(np.uint8)
        #DRAM Access Time
        latency = self.get_transfer_latency(lut_size)
        wait_start_time = max(self.sim_time, self.lut_valid_timer)
        if self.dma_timer > wait_start_time:
            self.lut_stat.wait_append(wait_start_time, self.dma_timer - wait_start_time)
        self.update_dma_timer(self.sim_time, latency)
        self.lut_valid_timer = self.dma_timer
        self.lut_stat.busy_append(self.dma_timer - latency, latency)
        self.logger.debug("LUT timer & dma_timer is updated to {}".format(self.dma_timer))

    def access_wmem(self):
        # Latency
        current_time = self.sim_time
        time_gap = math.ceil(max(0, self.wmem_valid_timer[self.wmem_in_use] - current_time))
        if time_gap > 0:
            self.logger.debug("Time {}: WMEM {} load delay occured & time_gap = {}".format(current_time, self.wmem_in_use, time_gap))
            self.wmem_stat[self.wmem_in_use].delay_append(current_time, time_gap)
            self.wmem_valid_timer[self.wmem_in_use] = -1
        return math.ceil(time_gap)

    def access_fmem(self, bank_idx):
        current_time = self.sim_time
        time_gap = math.ceil(max(0, self.fmem_valid_timer[bank_idx] - current_time))
        if time_gap > 0:
            self.fmem_stat[bank_idx].delay_append(current_time, time_gap)
            self.fmem_valid_timer[bank_idx] = -1
            self.logger.debug("Time {}: FMEM {} load delay occured & time_gap = {}".format(current_time, bank_idx, time_gap))
        return math.ceil(time_gap)

    def access_bmem(self):
        current_time = self.sim_time
        time_gap = math.ceil(max(0, self.bmem_valid_timer[self.bmem_in_use] - current_time))
        if time_gap > 0:
            self.bmem_stat[self.bmem_in_use].delay_append(current_time, time_gap)
            self.bmem_valid_timer[self.bmem_in_use] = -1
            self.logger.debug("Time {}: BMEM {} load delay occured & time_gap = {}".format(current_time, self.bmem_in_use, time_gap))
        return math.ceil(time_gap)
    
    def access_tmem(self):
        current_time = self.sim_time
        time_gap = math.ceil(max(0, self.tmem_valid_timer[self.tmem_in_use] - current_time))
        if time_gap > 0:
            self.tmem_stat[self.tmem_in_use].delay_append(current_time, time_gap)
            self.tmem_valid_timer[self.tmem_in_use] = -1
            self.logger.debug("Time {}: TMEM {} load delay occured & time_gap = {}".format(current_time, self.tmem_in_use, time_gap))
        return math.ceil(time_gap)
    
    def access_lut(self):
        current_time = self.sim_time
        time_gap = math.ceil(max(0, self.lut_valid_timer - current_time))
        if time_gap > 0:
            self.lut_stat.delay_append(current_time, time_gap)
            self.lut_valid_timer = -1
            self.logger.debug("Time {}: LUT load delay occured & time_gap = {}".format(current_time, time_gap))
        return math.ceil(time_gap)

    def update_dma_timer(self, pivot_time, delay):
        self.dma_timer = max(pivot_time, self.dma_timer) + delay

    def update_bmem_timer(self, pivot_time, delay):
        for bid in range(len(self.bmem_valid_timer)):
            if self.bmem_valid_timer[bid] > pivot_time:
                self.bmem_stat[bid].delay_transfer(pivot_time, delay)
                self.bmem_valid_timer[bid] += delay
    
    def update_wmem_timer(self, pivot_time, delay):
        for wid in range(len(self.wmem_valid_timer)):
            if self.wmem_valid_timer[wid] > pivot_time:
                self.wmem_stat[wid].delay_transfer(pivot_time, delay)
                self.wmem_valid_timer[wid] += delay
    
    def update_fmem_timer(self, pivot_time, delay):
        for fid in range(len(self.fmem_valid_timer)):
            if self.fmem_valid_timer[fid] > pivot_time:
                self.fmem_stat[fid].delay_transfer(pivot_time, delay)
                self.fmem_valid_timer[fid] += delay

    def load_host(self, data_name, size):
        #DRAM Access Time
        latency = self.get_transfer_latency(size)
        wait_start_time = max(self.sim_time, self.host_valid_timer)
        if self.dma_timer > wait_start_time:
            self.host_stat.wait_append(wait_start_time, self.dma_timer - wait_start_time)
        self.update_dma_timer(self.sim_time, latency)
        self.host_valid_timer = self.dma_timer
        self.host_stat.busy_append(self.dma_timer - latency, latency)
        self.logger.debug("LOAD host: host timer & dma_timer is updated to {}".format(self.host_valid_timer))
    
    def sync_host(self):
        current_time = self.sim_time
        time_gap = math.ceil(max(0, self.host_valid_timer - current_time))
        if time_gap > 0:
            self.host_stat.delay_append(current_time, time_gap)
            self.host_valid_timer = -1
            self.logger.debug("Time {}: Host load delay occured & time_gap = {}".format(current_time, time_gap))
        return math.ceil(time_gap)


class TVMemoryManager(VMemoryManager): # Use dump memory file test
    def __init__(self, manager):
        super().__init__(manager)
    
    def read_dram_data(self, name, offset, size):
        dt, address = self.dram_dict[name]
        dram_address = address + offset
        return self.dram_data[dt][dram_address:dram_address + size]

    def tmem_to_dram(
            self,
            data_name,
            dram_pivot_offset,
            transfer_unit_size,
            transfer_offset,
            num_transfers,
            tmem_pivot_address = 0,
            ):
        # Update timer
        write_start_time = max(self.sim_time, *self.tmem_valid_timer)
        write_delay = num_transfers * self.get_transfer_latency(transfer_unit_size)
        wait_start_time = max(self.sim_time, self.tmem_valid_timer[self.tmem_in_use])
        if write_start_time > wait_start_time:
            self.tmem_stat[self.tmem_in_use].wait_append(wait_start_time, write_start_time - wait_start_time)
        self.tmem_stat[self.tmem_in_use].busy_append(write_start_time, write_delay)
        self.tmem_valid_timer[self.tmem_in_use] = write_start_time + write_delay
        # Assumption: Highest Priority
        required_update_fn = [self.update_dma_timer, self.update_bmem_timer, self.update_wmem_timer, self.update_fmem_timer]
        for update_timer in required_update_fn:
            update_timer(write_start_time, write_delay)
        ##
        dt, address = self.dram_dict[data_name]
        dram_pivot_address = address + dram_pivot_offset
        for idx in range(num_transfers):
            dram_address = dram_pivot_address + transfer_offset * idx
            tmem_address = tmem_pivot_address + transfer_unit_size * idx
            self.dram_data[dt][dram_address:dram_address + transfer_unit_size] = \
                    self.tmem[self.tmem_in_use][tmem_address:tmem_address + transfer_unit_size]

    def write_host(self, data_name, data):
        size = data.size
        write_start_time = self.sim_time
        write_delay = self.get_transfer_latency(size)
        wait_start_time = max(self.sim_time, self.host_valid_timer)
        if write_start_time > self.sim_time:
            self.host_stat.wait_append(wait_start_time, write_start_time - wait_start_time)
        self.host_stat.busy_append(write_start_time, write_delay)
        self.host_valid_timer = write_start_time + write_delay
        # Assumption: Highest Priority
        required_update_fn = [self.update_dma_timer]
        for update_timer in required_update_fn:
            update_timer(write_start_time, write_delay)
        ##
        dt, address = self.dram_dict[data_name]
        dram_address = address
        self.dram_data[dt][dram_address:dram_address + size] = data
        self.logger.debug("WRITE host: host timer & dma_timer is updated to {}".format(self.host_valid_timer))


## TODO... for Keonjoo? : Make virtual estimator version for Dure-core DMA
