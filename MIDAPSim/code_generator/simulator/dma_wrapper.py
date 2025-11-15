from code_generator.simulator.memory import Memory
from code_generator.generator.memory_op_generator import TransferInfo
import logging

from config import cfg

from code_generator.env import rule
from code_generator.env.sfr import reg

class DMAWrapper():
    def __init__(self, simulator, config):
        self.simulator = simulator
        logging.config.dictConfig(cfg.LOGGING_CONFIG_DICT)
        self.logger = logging.getLogger('debug')
        self.sram_offset_info = rule.get_sram_offset_info(config)
        self.request_array = []
    
    @property
    def memory(self) -> Memory:
        return self.simulator.memory

    @property
    def registers(self):
        return self.simulator.registers

    @property
    def tid_offset(self):
        return self.sram_offset_info.tmem_id_offset

    @property
    def wid_offset(self):
        return self.sram_offset_info.wmem_id_offset
    
    @property
    def bid_offset(self):
        return self.sram_offset_info.bmem_id_offset

    @property
    def fid_offset(self):
        return self.sram_offset_info.fmem_id_offset
    
    def setup_request_array(self, request_array):
        self.request_array = request_array

    ## DMA Triggering
    def run(self):
        trigger, trigger_id = self.registers.get_value(reg.trigger_dma)
        tinfo = self.request_array[trigger_id]
        kwargs = {}
        kwargs['mem_id'] = tinfo.mem_id
        kwargs['transfer_unit_size'] = tinfo.transfer_unit_size
        kwargs['num_transfers'] = tinfo.num_transfers
        kwargs['dst_pivot_address'] = tinfo.dst_addr
        kwargs['src_pivot_address'] = tinfo.src_addr
        kwargs['dst_transfer_offset'] = tinfo.dst_transfer_offset
        kwargs['src_transfer_offset'] = tinfo.src_transfer_offset
        self.logger.debug("Transfer Info: {}".format(kwargs))
        self.registers.set(reg.ffd, trigger_id) # Read dma info fin
        ### Enable the busy-table & Check & Wait for available
        sbt = self.registers.get_value(reg.sbt)
        bitset_code = 1 << kwargs['mem_id']
        sbt = sbt | bitset_code
        self.registers.set(reg.sbt, sbt)
        f_sbt = self.registers.get_value(reg.f_sbt)
        # Instead of waiting... in real DMA Controller : It must be polling scheme
        if (bitset_code & f_sbt) > 0:
            self.logger.warning("Expected wait bits != 0, {}-th bit for reg f_sbt, {}".format(
                kwargs['mem_id'], f_sbt))
        ### Instead of Hardware-based triggering
            # Wait for PIPELINE DEPTH cycles to avoid conflict
        self.simulator.wait_datapath(rule.PIPELINE_DEPTH)
        ### 
        if (kwargs['dst_pivot_address'] & 0xF0000000) == 0x60000000:
            self.read_transfer(**kwargs)
        else:
            self.write_transfer(**kwargs)
        ## Disable busy table
        sbt = self.registers.get_value(reg.sbt)
        bitset_code = ~(1 << kwargs['mem_id'])
        sbt = sbt & bitset_code
        self.registers.set(reg.sbt, sbt)

   ############################ DMA Functions ################################
    def write_transfer(
            self,
            mem_id,
            transfer_unit_size,
            num_transfers,
            dst_pivot_address,
            src_pivot_address,
            dst_transfer_offset,
            src_transfer_offset,
            ):
        if mem_id >= self.wid_offset: # TODO: Allow for all memory units
            raise ValueError("Invalid")
        # TMEM - only
        _tmem_pivot_addr = src_pivot_address - self.sram_offset_info.tmem_pivot_address
        tmem_idx = _tmem_pivot_addr // self.sram_offset_info.tmem_addr_offset
        tmem_pivot_addr = _tmem_pivot_addr % self.sram_offset_info.tmem_addr_offset
        self.logger.debug("Transfer TMEM: from tmem[{0},{1}], {2}x{3} to DRAM Addr = {4}, {2}x{3}".format(
            tmem_idx, tmem_pivot_addr, transfer_unit_size, num_transfers, dst_pivot_address))
        for i in range(num_transfers):
            tmem_addr = tmem_pivot_addr + i * src_transfer_offset
            dram_addr = dst_pivot_address + i * dst_transfer_offset
            self.memory.write_dram_with_offset(
                    tmem_idx,
                    tmem_addr,
                    dram_addr,
                    transfer_unit_size
                    )
    
    def read_transfer(
            self,
            mem_id,
            transfer_unit_size,
            num_transfers,
            dst_pivot_address,
            src_pivot_address,
            dst_transfer_offset,
            src_transfer_offset,
            ):
        if mem_id - self.sram_offset_info.bmem_id_offset in [0, 1]:
            if num_transfers > 1:
                raise ValueError("Invalid DMA Code generation")
            bmem_pivot_addr = dst_pivot_address - self.sram_offset_info.bmem_pivot_address
            bmem_idx = bmem_pivot_addr // self.sram_offset_info.bmem_addr_offset
            bmem_addr = bmem_pivot_addr % self.sram_offset_info.bmem_addr_offset
            self.memory.load_bmem(bmem_idx, bmem_addr, src_pivot_address, transfer_unit_size)
        elif mem_id - self.sram_offset_info.wmem_id_offset in [0, 1]:
            wmem_pivot_addr = dst_pivot_address - self.sram_offset_info.wmem_pivot_address
            for i in range(num_transfers):
                waddr = wmem_pivot_addr + i * dst_transfer_offset
                dram_addr = src_pivot_address + i * src_transfer_offset
                wmem_idx = waddr // self.sram_offset_info.wmem_addr_offset
                wmem_addr = waddr % self.sram_offset_info.wmem_addr_offset
                self.memory.load_wmem(wmem_idx, wmem_addr, dram_addr, transfer_unit_size)
        elif mem_id - self.sram_offset_info.lut_id_offset in [0]:
            if num_transfers > 1:
                raise ValueError("Invalid DMA Code generation")
            lut_pivot_addr = dst_pivot_address - self.sram_offset_info.lut_pivot_address
            lut_addr = lut_pivot_addr
            self.memory.load_lut(lut_addr, src_pivot_address, transfer_unit_size)
        elif mem_id - self.sram_offset_info.fmem_id_offset in list(range(self.memory.fmem.shape[0])):
            if num_transfers > 1:
                raise ValueError("Invalid DMA Code generation")
            fmem_pivot_addr = dst_pivot_address - self.sram_offset_info.fmem_pivot_address
            fmem_idx = fmem_pivot_addr // self.sram_offset_info.fmem_addr_offset
            fmem_addr = fmem_pivot_addr % self.sram_offset_info.fmem_addr_offset
            if fmem_addr > 0:
                raise ValueError("Invalid DMA Code generation")
            self.memory.load_fmem(fmem_idx, src_pivot_address, transfer_unit_size)
        else:
            raise ValueError("Invalid DMA Code Generation(src pivot address)")


class DMA3DWrapper(DMAWrapper):
    def __init__(self, simulator, config):
        super().__init__(simulator, config)
        self.rdma_request_array = []
        self.wdma_request_array = []
        self.rdma_tid = 0
        self.wdma_working = 0
        self.wdma_triggered = -1
        self.write_used_space = 0

    @property
    def write_available_space(self):
        return self.memory.write_available_space - self.write_used_space

    def setup_request_array(self, request_array):
        self.rdma_request_array, self.wdma_request_array = request_array
        self.write_used_space = self.wdma_request_array[self.wdma_triggered + 1].data_size if self.wdma_request_array and len(self.wdma_request_array) > self.wdma_triggered + 1 else 0

    def run(self):
        # Update WDMA-related SFRs
        trigger, trigger_id = self.registers.get_value(reg.trigger_wdma)
        if self.wdma_triggered != trigger_id:
            self.wdma_triggered = trigger_id
            if trigger_id < len(self.wdma_request_array) - 1:
                descriptor = self.wdma_request_array[trigger_id + 1]
                while self.write_available_space < descriptor.data_size:
                    self._process_descriptor()
                self.write_used_space += descriptor.data_size
                assert self.write_available_space >= 0
            self.registers.set(reg.ffw, trigger_id + 1)
        # Update RDMA-related SFRs
        trigger, trigger_id = self.registers.get_value(reg.trigger_dma)
        if self.rdma_tid != trigger_id:
            self.rdma_tid = trigger_id
            descriptor = self.rdma_request_array[trigger_id]
            self.registers.set(reg.ffd, trigger_id) # Read dma info fin
            sbt = self.registers.get_value(reg.sbt)
            bitset_code = 1 << descriptor.mem_id
            sbt = sbt | bitset_code
            self.registers.set(reg.sbt, sbt)
            f_sbt = self.registers.get_value(reg.f_sbt)
            # Instead of waiting... in real DMA Controller : It must be polling scheme
            if (bitset_code & f_sbt) > 0:
                self.logger.warning("Expected wait bits != 0, {}-th bit for reg f_sbt, {}".format(
                    descriptor.mem_id, f_sbt))
            ### Instead of Hardware-based triggering
                # Wait for PIPELINE DEPTH cycles to avoid conflict
            self.simulator.wait_datapath(rule.PIPELINE_DEPTH)
            self.read_transfer_3D(
                mem_id=descriptor.mem_id,
                ddr_aoff=0,
                ddr_acnt=descriptor.acnt*descriptor.bcnt,
                baddr_ddr=descriptor.baddr_ddr_offset,
                baddr_sram=descriptor.baddr_sram,
                sram_boff=descriptor.boff,
                sram_acnt=descriptor.acnt
            )
            ## Disable busy table
            sbt = self.registers.get_value(reg.sbt)
            bitset_code = ~(1 << descriptor.mem_id)
            sbt = sbt & bitset_code
            self.registers.set(reg.sbt, sbt)

    def write_transfer_3D(
        self,
        aoff: int,
        acnt: int,
        boff: int,
        bcnt: int,
        coff: int,
        ccnt: int,
        csub: bool,
        bsub: bool,
        bcsync: bool,
        baddr_ddr: int
    ):
        dram_addr = baddr_ddr + aoff
        for i in range(ccnt):
            for j in range(bcnt):
                self.memory.write_dram_with_offset(0, 0, dram_addr, acnt)
                dram_addr += boff * (-1) ** int(bsub)
            dram_addr += coff * (-1) ** int(csub) - boff * (bcnt if bcsync else 1) * (-1) ** int(bsub)
        self.write_used_space -= acnt * bcnt * ccnt

    def read_transfer_3D(
        self,
        mem_id,
        ddr_aoff: int,
        ddr_acnt: int,
        baddr_ddr: int,
        baddr_sram: int,
        sram_boff: int,
        sram_acnt: int
    ):
        self.read_transfer(
            mem_id,
            sram_acnt,
            ddr_acnt // sram_acnt,
            baddr_sram,
            baddr_ddr + ddr_aoff,
            sram_boff,
            sram_acnt
        )

    def _process_descriptor(self):
        self.wdma_working += 1
        working_descriptor = self.wdma_request_array[self.wdma_working]
        self.write_transfer_3D(
            aoff=0,
            acnt=working_descriptor.acnt,
            boff=working_descriptor.boff,
            bcnt=working_descriptor.bcnt,
            coff=working_descriptor.coff,
            ccnt=working_descriptor.ccnt,
            csub=working_descriptor.csub,
            bsub=working_descriptor.bsub,
            bcsync=working_descriptor.bcsync,
            baddr_ddr=working_descriptor.baddr_ddr_offset
        )

    def wait_last_transfer(self):
        while self.write_used_space > 0:
            self._process_descriptor()
