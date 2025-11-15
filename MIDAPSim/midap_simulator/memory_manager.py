from __future__ import absolute_import, division, print_function, unicode_literals

import os
import math
import logging

import numpy as np

from config import cfg

import matplotlib.pyplot as plt


class MemoryStat():
    def __init__(self):
        self.busy = []
        self.delay = []
        self.wait = []

    def busy_append(self, start, cycles):
        self.busy.append((math.ceil(start), math.ceil(cycles)))

    def delay_append(self, start, cycles):
        self.delay.append((math.ceil(start), math.ceil(cycles)))

    def wait_append(self, start, cycles):
        overlap_time = max(0, self.wait[-1][0] + self.wait[-1][1] - start) if self.wait else 0
        if overlap_time > 0:
            start += overlap_time
            cycles -= overlap_time
        self.wait.append((math.ceil(start), math.ceil(cycles)))

    def delay_transfer(self, pivot_time, delay):
        if self.wait:
            pivot_time = math.ceil(max(pivot_time, self.wait[-1][0] + self.wait[-1][1]))
            delay = math.ceil(delay)
        for i in reversed(range(len(self.busy))):
            item = self.busy[i]
            if item[0] + item[1] <= pivot_time:
                # This task has already been finished
                break
            elif item[0] >= pivot_time:
                # Delay execution
                self.busy[i] = (item[0] + delay, item[1])
            else:
                # Stall until the DMA is ready
                delayed_cycle = item[1] - (pivot_time - item[0])
                self.busy[i] = (item[0], item[1] - delayed_cycle)
                self.busy.insert(i + 1, (pivot_time + delay, delayed_cycle))
                break
        self.wait.append((pivot_time, delay))


class MemoryManager():
    def __init__(self, manager):
        # Set FMEM constraints
        self.manager = manager
        self.config = manager.config
        self.num_fmem = self.config.MIDAP.FMEM.NUM
        self.fmem_size = self.config.MIDAP.FMEM.NUM_ENTRIES
        self.system_width = self.config.MIDAP.SYSTEM_WIDTH
        # FMEM 
        
        # Set WMEM constraints
        self.num_wmem = self.config.MIDAP.WMEM.NUM
        self.wmem_size = self.config.MIDAP.WMEM.NUM_ENTRIES
        self.ewmem_size = self.config.MIDAP.WMEM.E_NUM_ENTRIES
        # double buffered WMEM
        self.wmem_in_use = -1
        # BMEM setting
        self.bmem_size = self.config.MIDAP.BMEM.NUM_ENTRIES
        # double buffered BMEM
        self.bmem_in_use = -1
        # Double buffered Temporal Write Buffer memory 
        self.tmem_in_use = 0
        
        if(self.config.MODEL.QUANTIZED):
            data_type = np.int8
            fill = 0
            self.fmem = np.full([self.num_fmem, self.fmem_size], fill, dtype = data_type)
            self.wmem = np.full([2, self.num_wmem, self.ewmem_size], fill, dtype = data_type)
            self.bmem = np.full([2, self.bmem_size], fill, dtype = data_type)
            self.tmem = np.full([2, self.config.MIDAP.WRITE_BUFFER.NUM_ENTRIES], fill, dtype = data_type)
            self.lut_raw  = np.full([self.config.MIDAP.LUT.NUM_ENTRIES * 2 * 2], fill, dtype = np.uint8)
        else:
            data_type = np.float32
            self.fmem = np.zeros([self.num_fmem, self.fmem_size], dtype = data_type)
            self.wmem = np.zeros([2, self.num_wmem, self.ewmem_size], dtype = data_type)
            self.bmem = np.zeros([2, self.bmem_size], dtype = data_type)
            self.tmem = np.zeros([2, self.config.MIDAP.WRITE_BUFFER.NUM_ENTRIES], dtype = data_type)
            self.lut_raw  = None
        #
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('debug')
        self.dram_data = None
        self.dram_dict = None
        # type
        self.dram_offset = [
            self.config.DRAM.OFFSET.SHARED,
            self.config.DRAM.OFFSET.INPUT,
            self.config.DRAM.OFFSET.OUTPUT,
            self.config.DRAM.OFFSET.WEIGHT_BIAS,
            self.config.DRAM.OFFSET.BUFFER,
        ]
        self.data_size = 4 if not self.config.MODEL.QUANTIZED else 1 # byte per data
        self.core_id = self.config.MIDAP.CORE_ID
        # Memory status
        self.fmem_stat = [MemoryStat() for _ in range(self.num_fmem)]
        self.wmem_stat = [MemoryStat(), MemoryStat()]
        self.tmem_stat = [MemoryStat(), MemoryStat()]
        self.bmem_stat = [MemoryStat(), MemoryStat()]
        self.lut_stat = MemoryStat()
        self.host_stat = MemoryStat()
        self.write_stat = MemoryStat()

    def reset_sram(self):
        self.wmem_in_use = -1
        self.bmem_in_use = -1
        self.tmem_in_use = 0
        self.fmem[:, :] = 0
        self.wmem[:, :, :] = 0
        self.bmem[:, :] = 0
        self.tmem[:, :] = 0
        if self.lut_raw is not None:
            self.lut_raw[:] = 0

    @property
    def sim_time(self):
        return self.manager.stats.total_cycle()

    @property
    def lut(self):
        return self.lut_raw.reshape(self.config.MIDAP.LUT.NUM_ENTRIES, -1)

    def set_dram_info(self, dram_data, dram_dict):
        self.dram_data = dram_data
        self.dram_dict = dram_dict
    
    def store(self):
        return None

    def restore(self, save):
        pass

    def sync(self, sync_id):
        pass
    
    def wait_sync(self, wait_id):
        return 0

    def switch_wmem(self):
        self.wmem_in_use = (self.wmem_in_use + 1) % 2
        self.logger.debug("Switch_WMEM: to "+str(self.wmem_in_use))
    
    def switch_bmem(self):
        self.bmem_in_use = (self.bmem_in_use + 1) % 2
        self.logger.debug("Switch_BMEM: to "+str(self.bmem_in_use))

    def reset_wmem(self):
        self.wmem_in_use = -1

    def load_wmem(self, to_all, filter_name, filter_size = 0, filter_offset = 0, wmem_offset = 0, dram_offset = 0):
        pass

    def load_fmem(self, fmem_idx, data_name, data_size, fmem_offset = 0, dram_offset = 0):
        pass

    def load_bmem(self, bias_name, bias_size):
        pass
    
    def load_lut(self, lut_name):
        pass

    def access_wmem(self):
        return 0

    def access_fmem(self):
        return 0

    def access_bmem(self):
        return 0
    
    def access_lut(self):
        return 0

    def read_wmem(self, buf, extended_cim, address):
        time_gap = self.access_wmem()
        if not extended_cim and address + self.system_width > self.wmem_size:
            self.logger.error("WMEM Size: {} vs Requested Address: {}".format(self.wmem_size, address + self.system_width))
            raise ValueError("Wrong Address")
        if address + self.system_width > self.ewmem_size:
            self.logger.error("Extended WMEM Size: {} vs Requested Address: {}".format(self.ewmem_size, address + self.system_width))
            raise ValueError("Wrong Address")
        data_set_size = 1 if extended_cim else self.num_wmem
        sw = self.system_width
        address = (address // sw) * sw
        try:
            buf[:data_set_size,:self.system_width] = \
                    self.wmem[self.wmem_in_use, :data_set_size, address:address+self.system_width]
        except Exception:
            print("WMEM Read Error: " + str(address))
        return time_gap

    def read_fmem(self, buf, bank_idx, address):
        time_gap = self.access_fmem(bank_idx)
        sw = self.system_width
        address = (address // sw) * sw
        try:
            buf[:self.system_width] = self.fmem[bank_idx, address:address+self.system_width]
        except Exception:
            print("FMEM Read Error: bank {}, addr {}".format(bank_idx, address))
        return time_gap

    def read_bmem(self, buf, address):
        time_gap = self.access_bmem()
        sw = self.system_width
        address = (address // sw) * sw
        buf[:sw] = self.bmem[self.bmem_in_use, address:address + sw]
        return time_gap
    
    def write_fmem(self, bank_idx, address, data, valid):
        time_gap = self.access_fmem(bank_idx)
        if time_gap > 0:
            raise ValueError("Write must be detached from prefetching.. check compile result")
        sw = self.system_width
        address = (address // sw) * sw
        for i, flag in enumerate(valid):
            if flag:
                self.fmem[bank_idx, address + i * self.num_wmem : address + (i+1) * self.num_wmem] = \
                        data[i*self.num_wmem : (i+1) * self.num_wmem]
        return 0

    def write_tmem(self, address, data, valid): # 
        time_gap = self.access_tmem()
        sw = self.system_width
        address = (address // sw) * sw
        for i, flag in enumerate(valid):
            if flag:
                self.tmem[self.tmem_in_use, address + i * self.num_wmem : address + (i+1) * self.num_wmem] = \
                        data[i*self.num_wmem : (i+1) * self.num_wmem]
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
        pass

    def switch_tmem(self):
        self.tmem_in_use = (self.tmem_in_use + 1) % 2
        self.logger.debug("Switch_TMEM: to "+str(self.tmem_in_use))

    def write_dram(self, data_name, address, data, offset, size): 
        return 0

    def load_host(self, data_name, size):
        pass

    def write_host(self, data_name, data):
        pass

    def sync_host(self):
        return 0

    def show_timeline(self, path_info, block = True):
        fig, gnt = plt.subplots()
        gnt.set_xlabel("Cycles")
        gnt.set_ylabel("Component")
        component = [('WMEM_' + str(i), self.wmem_stat[i]) for i in range(len(self.wmem_stat))] + \
            [('FMEM_' + str(i), self.fmem_stat[i]) for i in range(len(self.fmem_stat))] + \
            [('TMEM_' + str(i), self.tmem_stat[i]) for i in range(len(self.tmem_stat))] + \
            [('BMEM_' + str(i), self.bmem_stat[i]) for i in range(len(self.bmem_stat))] + \
            [("LUT", self.lut_stat), ("HOST", self.host_stat), ("WRITE", self.write_stat)]
        yticks = [5 + 10 * i for i in range(len(component) + 2)]
        gnt.set_yticks(yticks)
        gnt.set_yticklabels(['CORE IDLE', 'DMA BUSY'] + [item[0] for item in reversed(component)])
        xticks = []
        xtickslabels = []
        current_cycle = 0
        for layer in path_info:
            xticks.append(current_cycle)
            xtickslabels.append(layer.name + '     ')
            current_cycle += self.manager.stats.layer_stats[layer.name]["CLOCK"]
        xticks[0] = 1
        xticks.append(current_cycle)
        xtickslabels.append(' ')
        gnt.set_xticks(xticks, labels=xtickslabels, minor=True, rotation='vertical')
        gnt.grid(True, axis='y')
        gnt.grid(True, axis='x', which='minor')
        for i in range(len(component)):
            gnt.broken_barh(component[i][1].wait, (yticks[-i-1], 4), facecolors=('cyan'))
            gnt.broken_barh(component[i][1].busy, (yticks[-i-1], 4), facecolors=('tab:blue'))
            gnt.broken_barh(component[i][1].delay, (yticks[-i-1] - 4, 4), facecolors=('tab:red'))
            gnt.broken_barh(component[i][1].busy, (yticks[1] - 4, 8), facecolors=('tab:blue'))
            gnt.broken_barh(component[i][1].delay, (yticks[0] - 4, 8), facecolors=('tab:red'))
            if i == 0:
                gnt.legend(['Wait for DMA', 'DMA Busy', 'Core Idle'], loc='upper left')
        os.makedirs(self.config.MIDAP.PLOT_DIRECTORY, exist_ok=True)
        fig_width = min(640, math.ceil(self.manager.stats.global_stats.CLOCK
                                      / min(filter(lambda x: x > 0, [self.manager.stats.layer_stats[layer.name]["CLOCK"] for layer in path_info]))) // 10)
        plt.gcf().set_size_inches(fig_width, 8)
        plt.title(self.manager.model_name + '_core_' + str(self.core_id), loc='left')
        plt.tight_layout()
        plt.pause(0.001)
        plt.savefig(os.path.join(self.config.MIDAP.PLOT_DIRECTORY, self.manager.model_name + '_core_' + str(self.core_id) + '.svg'), format='svg')
        plt.show(block=block)
