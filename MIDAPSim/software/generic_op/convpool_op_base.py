from __future__ import annotations

from past.builtins import basestring

from .operator_base import OperatorBase


class ConvPoolOpBase(OperatorBase):
    def __init__(self, kernel=1, stride=1, pad=0, in_plane=False, **kwargs):
        super(ConvPoolOpBase, self).__init__(**kwargs)
        if isinstance(kernel, list) or isinstance(kernel, tuple):
            if len(kernel) != 2:
                raise ValueError("kernel with int value or (int, int) format is only supported")
            self.k_w, self.k_h = kernel
        else:
            self.k_w, self.k_h = kernel, kernel
        self.__stride = stride
        if isinstance(pad, list) or isinstance(pad, tuple):
            if len(pad) not in [2, 4]: 
                raise ValueError("pad with int value or (pad_h, pad_w) = int, int or (pad_t, pad_b, pad_l, pad_r) - int, int ,int ,int format is only supported")
            if len(pad) == 2:
                pad = [pad[0], pad[0], pad[1], pad[1]]
        elif isinstance(pad, basestring):
            if pad == "VALID":
                pad = [0, 0, 0, 0]
            else:
                raise ValueError("not supported type for padding")
        else:
            pad = [pad for _ in range(4)]
        self.pad_l, self.pad_r, self.pad_t, self.pad_b = pad
        self.reversed = False
        self._in_plane = in_plane
    
    @property
    def kernel(self):
        return (self.k_h, self.k_w)

    @property
    def pad(self):
        return (self.pad_t, self.pad_b, self.pad_l, self.pad_r)

    @property
    def stride(self):
        return self.__stride

    def flip_operation(self, flip):
        if self.reversed ^ flip:
            self.pad_r, self.pad_l = self.pad_l, self.pad_r
            self.reversed = not self.reversed

    def __repr__(self):
        options = "kernel(h,w): {}\tstride: {}\tpad: {}\t".format(
            [self.k_h, self.k_w],
            self.stride,
            [self.pad_t, self.pad_b, self.pad_l, self.pad_r],
        )
        return super(ConvPoolOpBase, self).__repr__() + options

    @property
    def in_plane(self):
        return self._in_plane
