from models import ModelBuilder
from models.efficientnet.efficientnet import efficientnet_effunet

def eff_unet(net = 'efficientnet-b0', input_size = 576, out_channels = 2, decompose = True):
    mb : ModelBuilder
    mb, encoder_outputs = efficientnet_effunet(net=net, input_channels = input_size)
    x = encoder_outputs[0][0]
    print("Encoder Finished")
    for idx in range(len(encoder_outputs) - 1):
        in_channel = encoder_outputs[idx][1]
        prev_feature, out_channel = encoder_outputs[idx+1]
        print(f"Decode for {idx} blk w/ {prev_feature}")
        x = mb.ConvTranspose(x, in_channel, in_channel - out_channel, 2, 2)
        if not decompose:
            x = mb.Concat([prev_feature, x])
            x = mb.Conv(x, in_channel, out_channel, 3, 1, 'same')
        else:
            x1 = mb.Conv(prev_feature, out_channel, out_channel, 3, 1, 'same', activation='Linear')
            x2 = mb.Conv(x, in_channel - out_channel, out_channel, 3, 1, 'same', activation='Linear')
            x = mb.Sum(x1, x2)
        x = mb.Conv(x, out_channel, out_channel, 3, 1, 'same')
    x = mb.Conv(x, encoder_outputs[-1][1], out_channels, 1, 1)
    return mb