from models import ModelBuilder
from models.mobilenet import mobilenet

def mobilenet_ssd(input_size=300, num_classes = 80):
    out_feature_channels = 6 * (num_classes + 4)
    mb, srcs = mobilenet(input_size, True)
    l11, l13 = srcs
    l11_out = mb.Conv(l11, 512, out_feature_channels, 1, 1)
    l13_out = mb.Conv(l13, 1024, out_feature_channels, 1, 1)
    ssd0 = mb.Conv(l13, 1024, 256, 1, 1)
    ssd0 = mb.Conv(ssd0, 256, 512, 3, 2, 1)
    ssd0_out = mb.Conv(ssd0, 512, out_feature_channels, 1, 1)
    ssd1 = mb.Conv(ssd0, 512, 128, 1, 1)
    ssd1 = mb.Conv(ssd1, 128, 256, 3, 2, 1)
    ssd1_out = mb.Conv(ssd1, 256, out_feature_channels, 1, 1)
    ssd2 = mb.Conv(ssd1, 256, 128, 1, 1)
    ssd2 = mb.Conv(ssd2, 128, 256, 3, 2, 1)
    ssd2_out = mb.Conv(ssd2, 256, out_feature_channels, 1, 1)
    ssd3 = mb.Conv(ssd2, 256, 128, 1, 1)
    ssd3 = mb.Conv(ssd3, 128, 256, 3, 2, 1)
    ssd3_out = mb.Conv(ssd3, 256, out_feature_channels, 1, 1)
    return mb
    