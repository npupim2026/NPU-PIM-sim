import numpy as np

# In actual, address module is a hard-wired logic
class AddressModule():
    def __init__(self, num_offsets = 4):
        self.offsets = np.zeros(num_offsets, dtype = np.uint32)

    def set(self, setup_signal, setup_offset):
        assert len(setup_signal) == len(setup_offset)
        assert len(setup_signal) == self.offsets.size
        # setup_signal: 0(ignore) 1(set), 2(add), 3(reset) --> reset signal is optional... TBD
        for idx, (signal, offset) in enumerate(zip(setup_signal, setup_offset)):
            if signal == 0:
                pass
            elif signal == 1:
                self.offsets[idx] = offset
            elif signal == 2:
                self.offsets[idx] += offset
            elif signal == 3:
                self.offsets[idx] = 0
            else:
                raise ValueError("Invalid signal: {}".format(setup_signal))
    @property
    def address(self):
        return np.sum(self.offsets)
