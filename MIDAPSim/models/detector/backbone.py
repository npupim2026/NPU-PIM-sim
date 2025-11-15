from models import ModelBuilder

def cspdarknet53_tiny(mb, input_layer):
    x = mb.Conv(input_layer, 3, 32, 3, 2, 'same', activation = 'Relu')
    x = mb.Conv(x, 32, 64, 3, 2, 'same', activation = 'Relu')
    _, x = csp_block(mb, x, 32)
    _, x = csp_block(mb, x, 64)
    route, x = csp_block(mb, x, 128)
    x = mb.Conv(x, 512, 512, 3, 1, 'same', activation = 'Relu')
    return route, x

def csp_block(mb, x, c):
    x = mb.Conv(x, 2*c, 2*c, 3, 1, 'same', activation = 'Relu')
    x1 = mb.Conv(x, 2*c, c, 3, 1, 'same', activation = 'Relu') # Routegroup 2, 1 -> Conv
    x1_1 = mb.Conv(x1, c, c, 3, 1, 'same', activation = 'Relu') # Conv
    x1_1 = mb.Conv(x1_1, c, 2*c, 3, 1, 'same') # Concat -> Conv 1 (Psum1)
    x1_2 = mb.Conv(x1, c, 2*c, 3, 1, 'same') # Concat -> Conv 2 (Psum2)
    route = mb.Sum(x1_1, x1_2, activation = 'Relu') # Concat -> Convolution (Psum1 + Psum2) # 2 * c
    x1 = mb.MaxPool(route, 2, 2)
    x2 = mb.MaxPool(x, 2, 2)
    x = mb.Concat([x2, x1]) # 4 * c
    return route, x