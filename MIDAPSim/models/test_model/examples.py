from models.model_builder import ModelBuilder

def dilation_test(input_shape=(1, 320, 32, 32)):
    # 2016 SKT disco GAN
    mb = ModelBuilder("dilation_test")
    x = mb.set_input_tensor(tensor_shape=input_shape)
    x = mb.DWConv(x, 320, 3, 1)
    x1 = mb.Conv(x, 320, 256, 3, 1, 6, 6)
    x2 = mb.Conv(x, 320, 256, 3, 1, 12, 12)
    x3 = mb.Conv(x, 320, 256, 3, 1, 18, 18)
    return mb

def bilinear_test(input_shape=(1, 256, 32, 32)):
    # 2016 SKT disco GAN
    mb = ModelBuilder("bilinear_test")
    x = mb.set_input_tensor(tensor_shape=input_shape)
    x1 = mb.Conv(x, 256, 256, 3, 2, 1)
    x2 = mb.UpsampleBilinear(x1, 2)
    x3 = mb.UpsampleBilinear(x1, 4)
    return mb