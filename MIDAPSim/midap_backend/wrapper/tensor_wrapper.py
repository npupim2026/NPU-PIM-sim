from software.network.types import VTensorType

class TensorWrapper(object):
    mapping_algorithms = ['default', 'linear', 'valid']
    def __init__(self, **kwargs):
        self.name = None
        self.orig_shape = None
        self.init_shape = None
        self.shape = None
        self.scale = (1, 1, 1)
        self.offset = (0, 0, 0)
        self.mapping_type = 'default'
        self.flip_x = False
        self.fixed = False

    def set_tensor(self, name, shape, orig_shape=None, init_shape=None, mapping_type='default', offset=None, scale=None, flip_x = False, **kwargs):
        type2algo = {
            VTensorType.Default : 'default',
            VTensorType.InputLinear: 'linear',
            VTensorType.InputValid: 'valid',
            VTensorType.OutputLinear: 'linear',
            VTensorType.OutputWMEM: 'wmem'
        }
        self.name = name
        self.shape = shape
        if orig_shape is None:
            orig_shape = shape
        if init_shape is None:
            init_shape = shape
        self.orig_shape = orig_shape
        self.init_shape = init_shape
        if mapping_type not in type2algo:
            raise ValueError("mapping_type should be one of {}".format(type2algo.keys()))
        self.mapping_type = type2algo[mapping_type]
        if offset is not None:
            self.offset = tuple(offset)
        if scale is not None:
            self.scale = tuple(scale)
        self.flip_x = flip_x

    def __repr__(self):
        return self.str_name() + self.str_algo() + self.str_shape() + '\n'

    def str_name(self):
        return "Tensor name: " + str(self.name)

    def str_algo(self):
        if self.mapping_type == 'default':
            return ''
        algo = "\nMapping algorithm: " + self.mapping_type
        algo_constant = "\nScale: " + str(self.scale) + " || Offset: " + str(self.offset)
        info = "\nOriginal shape: " + str(self.orig_shape)
        return algo + algo_constant + info

    def str_shape(self):
        return " || (Virtualized) Tensor shape: " + str(self.shape)
