from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from config import cfg

def im2col(input_data, main_op):
    # print(input_data.shape)
    _, W, H, C = input_data.shape
    input_data = input_data[0,:].transpose(2, 0, 1)  # WHC --> # C W H
    filter_w, filter_h = main_op.k_w, main_op.k_h
    stride = main_op.stride
    pad_t, pad_b, pad_l, pad_r = main_op.pad_t, max(0, main_op.pad_b), main_op.pad_l, max(0, main_op.pad_r)
    out_w = (W + pad_l + pad_r - filter_w) // stride + 1
    out_h = (H + pad_t + pad_b - filter_h) // stride + 1

    img = np.pad(input_data, [(0, 0), (pad_l, pad_r), (pad_t, pad_b)], 'constant')
    if main_op.pad_r < 0:
        img = img[:, :main_op.pad_r, :]
    if main_op.pad_b < 0:
        img = img[:, :, :main_op.pad_b]
    col = np.zeros((C, filter_w, filter_h, out_w, out_h))
    # C*W*H, outw*outh
    for x in range(filter_w):
        x_max = x + stride * out_w
        for y in range(filter_h):
            y_max = y + stride * out_h
            col[:, x, y, :, :] = img[:, x:x_max:stride, y:y_max:stride]

    col = col.reshape(1, -1, 1, out_w * out_h)  # CHW
    return col