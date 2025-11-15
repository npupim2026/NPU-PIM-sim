from midap_simulator.memory_controller import MemoryController


class Stage():
    def __init__(self, manager):
        self.manager = manager
        self.system_width = manager.config.MIDAP.SYSTEM_WIDTH
        self.num_wmem = manager.config.MIDAP.WMEM.NUM
        self.num_fmem = manager.config.MIDAP.FMEM.NUM
        self.memory_controller : MemoryController = manager.memory_controller
        self.skipped_pipeline_stages = 0
        self.initialize()
    
    def initialize(self):
        self.output_buf = None
        self.input_buf = None

    def set_input_buf(self, input_buf):
        self.input_buf = input_buf
    
    def setup(self, modules):
        pass

    def run(self, dataflow_info):
        pass
