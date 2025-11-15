from ..model_builder import ModelBuilder


def attention_head():
    mb = ModelBuilder('attention_head_test')
    D = mb.set_input_tensor(tensor_shape=(1, 768, 196, 1), name='D')
    V_w = mb.set_input_tensor(tensor_shape=(1, 768, 64, 1), name=f'V_w')
    Q = mb.MatMul([D], 196, 768, 64, name='Q')
    K = mb.MatMul([D], 196, 768, 64, name='K')
    V = mb.MatMul_binary_transpose(V_w, D, 64, 768, 196, name='V_T')
    A = mb.MatMul_binary_transpose(Q, K, 196, 64, 196, name='QK_T')
    A = mb.Concat([A], name='A')
    out = mb.MatMul_binary_transpose(A, V, 196, 196, 64, name='out')
    return mb

def matmul_test():
    mb = ModelBuilder('matmul_test')
    x = mb.set_input_tensor(tensor_shape=(1, 16, 196, 1))
    A = mb.Conv(x, 16, 196, 1, name='A')
    B = mb.Conv(x, 16, 192, 1, name='B')
    C = mb.MatMul([A, B], 196, 196, 192, name='C')
    return mb

def attention_block(mb, input, weight_V, block_name=''):
    input = mb.DummyNode(input)
    Q = mb.MatMul([input], 196, 768, 64, name=f'Q{block_name}')
    K = mb.MatMul([input], 196, 768, 64, name=f'K{block_name}')
    V = mb.MatMul_binary_transpose(weight_V, input, 64, 768, 196, name=f'V_T{block_name}')
    QK_T = mb.MatMul_binary_transpose(Q, K, 196, 64, 196, name=f'QK_T{block_name}')
    A = mb.Concat([QK_T], name=f'A{block_name}')   # Force off-chip write
    out = mb.MatMul_binary_transpose(A, V, 196, 196, 64, name=f'MHA_out{block_name}')
    return out

def multi_head_attention_test():
    mb = ModelBuilder('multi_head_attention')
    D = mb.set_input_tensor(tensor_shape=(1, 768, 196, 1), name='D')
    out = multi_head_attention(mb, D)
    return mb

def multi_head_attention(mb, input):
    return _multi_head_attention_v2(mb, input)

def _multi_head_attention_v1(mb, input):
    V_w = [mb.set_input_tensor(tensor_shape=(1, 768, 64, 1), name=f'V_w{i}') for i in range(12)]
    mha_out = attention_block(mb, input, V_w[0], 0)
    partial_sum = mb.MatMul([mha_out], 196, 64, 768, name=f'MatMul_out0')
    for i in range(1, 12):
        mha_out = attention_block(mb, input, V_w[i], i)
        mha_out = mb.MatMul([mha_out], 196, 64, 768, name=f'MatMul_out{i}')
        partial_sum = mb.Sum(partial_sum, mha_out)
    partial_sum = mb.Concat([partial_sum], name=f'layernorm')  # Force off-chip write
    x = mb.Sum(partial_sum, input, name='residual_connection')
    return x

def _multi_head_attention_v2(mb, input):
    V_w = [mb.set_input_tensor(tensor_shape=(1, 768, 64, 1), name=f'V_w{i}') for i in range(12)]
    mha_out = [attention_block(mb, input, V_w[i], i) for i in range(12)]
    concat = mb.Concat(mha_out, axis = 'c', name='MHA_out')
    matmul_out = mb.MatMul([concat], 196, 768, 768, name='MatMul_out')
    matmul_out = mb.Concat([matmul_out], name=f'layernorm') # Force off-chip write
    x = mb.Sum(matmul_out, input, name='residual_connection')
    return x

def _multi_head_attention_v3():
    mb = ModelBuilder('multi_head_attention')
    x = mb.set_input_tensor(tensor_shape=(1, 16, 256, 1))
    V_w = mb.set_input_tensor(tensor_shape=(1, 768, 64*12, 1), name='V_w')
    D_origin = mb.Conv(x, 16, 768, 1, name='D_origin')
    D = mb.Crop(D_origin, crop_y=[0, -60], name='D')
    # XXX: Temporal solution
    D_upsampled = mb.Upsample(D_origin, size=(256*12, 1))
    #K = []
    QK_T = []
    Q = mb.MatMul([D_upsampled], 256*12, 768, 64, name=f'Q')
    K = mb.MatMul([D_upsampled], 256*12, 768, 64, name=f'K')
    for i in range(12):
        #Q.append(mb.MatMul([D], 196, 768, 64, name=f'Q{i}'))
        #K.append(mb.MatMul([D], 196, 768, 64, name=f'K{i}'))
        Q_i = mb.Crop(Q, crop_y=[i * 256, (i-11) * 256], name=f'Q{i}')
        K_i = mb.Crop(K, crop_y=[i * 256, (i-11) * 256], name=f'K{i}')
        QK_T.append(mb.MatMul_binary_transpose(Q_i, K_i, 256, 64, 256, name=f'QK_T{i}'))
    QK_T_concat = mb.Concat(QK_T, axis='h', name='QK_T')
    #Q = mb.Concat([mb.MatMul([D], 196, 768, 64, name=f'Q{i}') for i in range(12)], axis='h')
    #K = mb.Concat([mb.MatMul()])
    V_trans_total = mb.MatMul_binary_transpose(V_w, D_origin, 64*12, 768, 256, name='V_T')
    # after softmax
    mha_temp = []
    for i in range(12):
        A = mb.Crop(QK_T_concat, crop_y=[i * 256, (i-11) * 256 - 60], name=f'A{i}')
        V_trans = mb.Crop(V_trans_total, crop_y=[i * 64, (i-11) * 64], name=f'V_T{i}')
        mha_temp.append(mb.MatMul_binary_transpose(A, V_trans, 196, 256, 64))
    concat = mb.Concat(mha_temp)
    #concat = mb.Concat([attention_block(mb, D, i+1) for i in range(12)])
    #mha_out = attention_block(mb, D, 1)
    #mha_out = mb.MatMul([mha_out], 196, 64, 768, name='MatMul_out1')
    #for i in range(3):#(11):
    #    mha_temp = attention_block(mb, D, i+2)
    #    mha_temp = mb.MatMul([mha_temp], 196, 64, 768, name=f'MatMul_out{i+2}')
    #    mha_out = mb.Sum(mha_out, mha_temp, name=f'partial_sum{i+1}')
    mha_out = mb.MatMul([concat], 196, 768, 768, name='MHA_out')
    x = mb.Sum(mha_out, D, name='residual_connection')
    return mb

def ffn(mb, input):
    x = mb.MatMul([input], 196, 768, 3072, activation='GELU', name='MatMul_FFN1')
    x = mb.MatMul([x], 196, 3072, 768, name='MatMul_FFN2')
    x = mb.Concat([x], name='layernorm2')
    x = mb.Sum(x, input, name='residual_connection2')
    return x

def vit_block_test():
    mb = ModelBuilder('vit_block_test')
    D = mb.set_input_tensor(tensor_shape=(1, 768, 196, 1), name='D')
    x = multi_head_attention(mb, D)
    x = ffn(mb, x)
    return mb
