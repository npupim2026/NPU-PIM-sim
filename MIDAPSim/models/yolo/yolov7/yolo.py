import os
import yaml
from models.model_builder import ModelBuilder
from .common import Conv, Concat, Upsample, Silence, ELANCat, ConvSumCat, MP, SP, SPPCSPC, RepConv#, Detect


def parse_model(mb, input, in_c, num_classes=None, cfg_path=os.path.join(os.path.dirname(__file__), 'cfg', 'yolov7.yaml')):
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    if num_classes is None:
        num_classes = cfg['nc']

    layers = [input]
    out_c = [in_c]
    for i, (f, n, m, args) in enumerate(cfg['backbone'] + cfg['head']):
        if m == 'nn.Upsample':
            m = 'Upsample'
        elif m == 'Detect':
            continue
        m = eval(m)
        if m in [Conv, SPPCSPC, RepConv]:
            args = [mb, layers[f], out_c[f]] + args
        elif m is ConvSumCat:
            args = [mb, list(map(lambda x: layers[x], f)), list(map(lambda x: out_c[x], f))] + args
        else:
            args = [mb, layers[f] if isinstance(f, int) else list(map(lambda x: layers[x], f))] + args
        if i == 0:
            layers = []
            out_c = []
        layers.append(m(*args))
        out_c.append(mb.model_dict[layers[-1]]['output'].shape[1])


def yolov7(input_size=640, num_classes=80):
    cfg_path = os.path.join(os.path.dirname(__file__), 'cfg', 'yolov7.yaml')
    mb = ModelBuilder(f'YOLOv7_{input_size}')
    input_shape = (1, 3, input_size, input_size)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    parse_model(mb, x, 3, num_classes, cfg_path=cfg_path)
    return mb

def yolov7_tiny(input_size=640, num_classes=80):
    cfg_path = os.path.join(os.path.dirname(__file__), 'cfg', 'yolov7-tiny.yaml')
    mb = ModelBuilder(f'YOLOv7-tiny_{input_size}')
    input_shape = (1, 3, input_size, input_size)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    parse_model(mb, x, 3, num_classes, cfg_path=cfg_path)
    return mb
