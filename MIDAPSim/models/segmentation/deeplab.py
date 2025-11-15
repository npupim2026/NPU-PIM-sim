from models import ModelBuilder
import models.mobilenet as mobilenet

def aspp(mb : ModelBuilder, x, backbone = 'mobilenet', input_size = 320, output_stride = 16):
    in_c = -1
    if backbone == 'mobilenet':
        in_c = 320
    else:
        raise NotImplementedError
    if output_stride == 16:
        dilations = [1, 6, 12, 18]
    elif output_stride == 8:
        dilations = [1, 12, 24, 36]
    else:
        raise NotImplementedError
    x1 = mb.Conv(x, in_c, 256, 1, pad=0, dilation=dilations[0])
    x2 = mb.Conv(x, in_c, 256, 3, pad=dilations[1], dilation=dilations[1])
    x3 = mb.Conv(x, in_c, 256, 3, pad=dilations[2], dilation=dilations[2])
    x4 = mb.Conv(x, in_c, 256, 3, pad=dilations[3], dilation=dilations[3])
    x5 = mb.GlobalPool(x)
    x5 = mb.Conv(x5, in_c, 256, 1)
    x5 = mb.Upsample(x5, scale=input_size//output_stride)
    x5 = mb.Conv(x5, 256, 256, 1, activation='Linear')
    x = mb.Concat([x1, x2, x3, x4])
    x = mb.Conv(x, 1024, 256, 1, activation='Linear')
    x = mb.Sum(x, x5, activation='Relu')
    return x

def decoder(mb : ModelBuilder, x, low_f, num_classes, backbone = 'mobilenet'):
    low_level_inplanes = -1
    if backbone == 'mobilenet':
        low_level_inplanes = 24
    l1 = mb.Conv(low_f, low_level_inplanes, 48, 1)
    x = mb.UpsampleBilinear(x, 4)
    x = mb.Concat([x, l1])
    x = mb.Conv(x, 304, 256, 3, 1, 1)
    x = mb.Conv(x, 256, 256, 3, 1, 1)
    x = mb.Conv(x, 256, num_classes, 1, 1, 0, activation='Linear')
    return x
    
def deeplab_v3(backbone = 'mobilenet', input_size=320, output_stride = 16, num_classes = 21):
    if backbone == 'mobilenet':
        mb, features = mobilenet.mobilenet_v2(input_size=input_size, output_stride=output_stride, backbone=True)
    else:
        raise NotImplementedError
    x = aspp(mb, features[-1], input_size=input_size, output_stride=output_stride, backbone=backbone)
    x = decoder(mb, x, features[0], num_classes, backbone=backbone)
    # x = mb.UpsampleBilinear(x, 4)
    return mb

## Cannot be supported...
def lite_raspp(
    mb : ModelBuilder,
    inputs,
    backbone = 'mobilenet_v3_small',
    input_shape = (224, 224),
    n_class = 2,
    avg_pool_kernel = (11, 11),
    avg_pool_strides = (4, 4)
    ):
    """LiteRASSP.
        # Arguments
            # init
                input_shape: Tuple/list of 2 integers, spatial shape of input
                    tensor
                n_class: Integer, number of classes.
                avg_pool_kernel: Tuple/integer, size of the kernel for
                    AveragePooling
                avg_pool_strides: Tuple/integer, stride for applying the of
                    AveragePooling operation
            # Call
                inputs: Tensor, input tensor of the model
                training: Mode for training-aware layers
    """
