from models import ModelBuilder
from .backbone import cspdarknet53_tiny

def yolov4_tiny(input_size=416, num_classes = 80):
    mb = ModelBuilder("yolov4_tiny_{}".format(input_size))
    x = mb.set_input_tensor(tensor_shape=(1, 3, input_size, input_size))
    route1, x = cspdarknet53_tiny(mb, x) # output channels = 512
    x = mb.Conv(x, 512, 256, 1, 1, 'same', activation='Relu')
    y = mb.Conv(x, 256, 512, 3, 1, 'same', activation='Relu')
    y0 = mb.Conv(y, 512, (num_classes + 5) * 3, 1, 1, 'same')
    x = mb.Conv(x, 256, 128, 3, 1, 'same')
    x1 = mb.Upsample(x, 2)
    x1 = mb.Conv(x1, 128, 256, 3, 1, 'same')
    x2 = mb.Conv(route1, 256, 256, 3, 1, 'same')
    x = mb.Sum(x1, x2, activation = 'Relu')
    y1 = mb.Conv(x, 256, (num_classes + 5) * 3, 1, 1, 'same')
    return mb
