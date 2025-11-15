from ..model_builder import ModelBuilder
import torch.nn as nn


layernorm_support = False


# Based on TorchVision implementation (https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py)
def ViT(
    image_size=224,
    patch_size=16,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
    num_classes=1000,
    representation_size=None
):
    assert image_size % patch_size == 0
    batch_size = (image_size // patch_size) * (image_size // patch_size) + 1

    mb = ModelBuilder(f'ViT_{image_size}')
    input_shape = (1, hidden_dim, batch_size, 1)
    x = mb.set_input_tensor(tensor_shape=input_shape)

    for i in range(num_layers):
        x = EncoderBlock(mb, x, num_heads, batch_size, hidden_dim, mlp_dim, block_idx=f'{i}')

    x = mb.Crop(x, crop_x=[0, 1], name='class_token')

    if representation_size is None:
        x = mb.FC(x, hidden_dim, num_classes)
    else:
        x = mb.FC(x, hidden_dim, representation_size, activation='sigmoid') # FIXME: 'tanh' in the original version
        x = mb.FC(x, representation_size, num_classes)

    return mb


def ViTLayer(
    image_size=224,
    patch_size=16,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
):
    assert image_size % patch_size == 0
    batch_size = (image_size // patch_size) * (image_size // patch_size) + 1

    mb = ModelBuilder(f'ViT_layer_{image_size}')
    input_shape = (1, hidden_dim, batch_size, 1)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    x = EncoderBlock(mb, x, num_heads, batch_size, hidden_dim, mlp_dim)

    return mb


def LayerNorm(mb, x, shape, name_postfix='', npu_support=False):
    if npu_support:
        return mb.LayerNorm(x, shape)
    if mb.model_dict[x].generic.is_first_tensor:
        return x
    out = mb.Concat([x], name='LayerNorm'+name_postfix)    # Baseline: Offload to the host
    out_tensor = mb.model_dict[out].output
    out_tensor = nn.LayerNorm(out_tensor.shape)(out_tensor) # XXX: Temporal solution just to compile
    mb.model_dict[out].output = out_tensor
    mb.model_dict[out].generic.output_tensor = out_tensor.detach().numpy()
    return out


def Softmax(mb, x, name_postfix=''):
    out = mb.Concat([x], name='LayerNorm'+name_postfix)    # Baseline: Offload to the host
    out_tensor = mb.model_dict[out].output
    out_tensor = nn.Softmax(1)(out_tensor)
    mb.model_dict[out].output = out_tensor
    mb.model_dict[out].generic.output_tensor = out_tensor.detach().numpy()
    return out


def EncoderBlock(mb, input, num_heads, batch_size, hidden_dim, mlp_dim, block_idx=''):
    x = LayerNorm(mb, input, hidden_dim, name_postfix=f'_E{block_idx}_1', npu_support=layernorm_support)
    x = MultiheadAttention(mb, x, num_heads, batch_size, hidden_dim, block_idx=block_idx)
    x = mb.Sum(x, input)
    y = LayerNorm(mb, x, hidden_dim, name_postfix=f'_E{block_idx}_2', npu_support=layernorm_support)
    y = MLP(mb, y, batch_size, hidden_dim, mlp_dim, block_idx=block_idx)
    return mb.Sum(y, x)


def MultiheadAttention(mb, input, num_heads, batch_size, hidden_dim, block_idx=''):
    #return _MultiheadAttention_baseline(mb, input, num_heads, batch_size, hidden_dim, block_idx)
    return _MultiheadAttention_optimized(mb, input, num_heads, batch_size, hidden_dim, block_idx)


def _MultiheadAttention_baseline(mb, input, num_heads, batch_size, hidden_dim, block_idx=''):
    head_dim = hidden_dim // num_heads
    V_w = [mb.set_input_tensor(tensor_shape=(1, hidden_dim, head_dim, 1), name=f'E{block_idx}_H{i}_V_w') for i in range(num_heads)]
    mha_out = []
    Q = mb.MatMul([input], batch_size, hidden_dim, hidden_dim, name=f'E{block_idx}_Q')
    Q = mb.Split(Q, head_dim, name=f'E{block_idx}_Q')
    for i in range(num_heads):
        K = mb.MatMul([input], batch_size, hidden_dim, head_dim, name=f'E{block_idx}_K{i}')
        V = mb.MatMul_binary_transpose(V_w[i], input, head_dim, hidden_dim, batch_size, name=f'E{block_idx}_V_T{i}')
        QK_T = mb.MatMul_binary_transpose(Q[i], K, batch_size, head_dim, batch_size, name=f'E{block_idx}_QK_T{i}')
        A = Softmax(mb, QK_T, name_postfix=f'_E{block_idx}_H{i}')
        mha_out.append(mb.MatMul_binary_transpose(A, V, batch_size, batch_size, head_dim, name=f'E{block_idx}_H{i}_out'))
    concat = mb.Concat(mha_out, axis='c')
    out = mb.MatMul([concat], batch_size, hidden_dim, hidden_dim, name=f'E{block_idx}_MHA_out')
    return out


def _MultiheadAttention_optimized(mb, input, num_heads, batch_size, hidden_dim, block_idx=''):
    head_dim = hidden_dim // num_heads
    input = [mb.DummyNode(input, name=f'E{block_idx}_H{i}_in') for i in range(num_heads)]
    V_w = [mb.set_input_tensor(tensor_shape=(1, hidden_dim, head_dim, 1), name=f'E{block_idx}_H{i}_V_w') for i in range(num_heads)]
    mha_out = []
    for i in range(num_heads):
        Q = mb.MatMul([input[i]], batch_size, hidden_dim, head_dim, name=f'E{block_idx}_Q{i}')
        K = mb.MatMul([input[i]], batch_size, hidden_dim, head_dim, name=f'E{block_idx}_K{i}')
        V = mb.MatMul_binary_transpose(V_w[i], input[i], head_dim, hidden_dim, batch_size, name=f'E{block_idx}_V_T{i}')
        QK_T = mb.MatMul_binary_transpose(Q, K, batch_size, head_dim, batch_size, name=f'E{block_idx}_QK_T{i}')
        A = Softmax(mb, QK_T, name_postfix=f'_E{block_idx}_H{i}')
        mha_out.append(mb.MatMul_binary_transpose(A, V, batch_size, batch_size, head_dim, name=f'E{block_idx}_H{i}_out'))
    concat = mb.Concat(mha_out, axis='c')
    out = mb.MatMul([concat], batch_size, hidden_dim, hidden_dim, name=f'E{block_idx}_MHA_out')
    return out


def MLP(mb, input, batch_size, hidden_dim, mlp_dim, block_idx=''):
    x = mb.MatMul([input], batch_size, hidden_dim, mlp_dim, activation='GELU', name=f'E{block_idx}_FFN_1')
    x = mb.MatMul([x], batch_size, mlp_dim, hidden_dim, name=f'E{block_idx}_FFN_2')
    return x
