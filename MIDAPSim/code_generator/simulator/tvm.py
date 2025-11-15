import numpy as np

# In actual, TVM is a hard-wired logic
class TVM(): # Tensor Virtualization Module
    def __init__(self):
        self.activate = 0
        self.scale_x = 0
        self.pivot_x = 0
        self.offset_x = 0
        self.scale_y = 0
        self.pivot_y = 0
        self.offset_y = [0, 0]
        self.y_counter = [0, 0]
        self.x_counter = 0

        # Parameters for replication padding
        self.rpad_activate = 0  # 4 bit
        self.rpad_l = 0         # 7 bit
        self.rpad_r = 0         # 7 bit
        self.rpad_t = 0         # 7 bit
        self.rpad_b = 0         # 7 bit
        self.rpad_x_counter = 0 # 8 bit
        self.rpad_y_counter = 0 # 8 bit

    @property
    def params_str(self):
        ret = f"Activate: {self.activate}, X_Params:{self.scale_x, self.pivot_x, self.offset_x}\n"
        ret += f"Y_Params: {self.scale_y, self.pivot_y, self.offset_y[0], self.offset_y[1]}"
        return ret

    def set(self, inputs): # reg.tvi1, reg.tvi2, reg.rpad
        if len(inputs) != 13:
            raise RuntimeError
        self.activate, self.scale_x, self.scale_y, self.pivot_x, \
             self.pivot_y, self.offset_y[0], self.offset_x, self.offset_y[1], \
             self.rpad_activate, self.rpad_l, self.rpad_r, self.rpad_t, self.rpad_b = inputs
        self.y_counter = [self.pivot_y, self.pivot_y]
        self.x_counter = self.pivot_x

    def process_signal(self, jump, signal):
        if self.activate == 0 and self.rpad_activate == 0:
            return signal
        new_signal = list(signal)
        # Replication padding must be applied before tensor virtualization logic
        if self.rpad_activate & 0x2 > 0:    # x axis
            if signal[3] == 2:
                if self.rpad_x_counter < self.rpad_l or self.rpad_x_counter >= self.rpad_r:
                    new_signal[3] = 0
                self.rpad_x_counter += 1
            elif signal[3] == 3:
                if jump == 1:
                    self.rpad_x_counter += 1
                else:
                    self.rpad_x_counter = 0
        if self.rpad_activate & 0x1 > 0:    # y axis
            if signal[4] == 2:
                if self.rpad_y_counter < self.rpad_t or self.rpad_y_counter >= self.rpad_b:
                    new_signal[4] = 0
                self.rpad_y_counter += 1
            elif signal[4] == 3:
                self.rpad_y_counter = 0
        if self.activate:
            if signal[1] == 2:
                self.y_counter[0] += self.offset_y[0]
                if self.y_counter[0] >= self.scale_y:
                    self.y_counter[0] = self.y_counter[0] - self.scale_y
                else:
                    new_signal[1] = 0
            elif signal[1] == 3:
                self.y_counter[0] = self.pivot_y
            if new_signal[3] == 2:
                self.x_counter += self.offset_x
                if self.x_counter >= self.scale_x:
                    self.x_counter = self.x_counter - self.scale_x
                else:
                    new_signal[3] = 0
            elif new_signal[3] == 3:
                self.x_counter = 0 if jump == 1 else self.pivot_x
            if new_signal[4] == 2:
                self.y_counter[1] += self.offset_y[1]
                if self.y_counter[1] >= self.scale_y:
                    self.y_counter[1] = self.y_counter[1] - self.scale_y
                else:
                    new_signal[4] = 0
            elif new_signal[4] == 3:
                self.y_counter[1] = self.y_counter[0]
        return new_signal
    
    def __repr__(self) -> str:
        return f"X: {self.x_counter}/{self.scale_x}, Y: {self.y_counter} / {self.scale_y}"
