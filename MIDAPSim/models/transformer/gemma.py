from ..model_builder import ModelBuilder
import torch.nn as nn
import json
import os
from software.generic_op import HostProcessOp


layernorm_support = False


# Based on TorchVision implementation (https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py)
def Gemma2(
    input_len=1,
    input_pos=0,
    vocab_size=256000,
    hidden_size=3072,
    intermediate_size=24576,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=256,
    max_seq_len=8192,
    hidden_activation="gelu_pytorch_tanh",
    max_position_embeddings=8192,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    pad_token_id=0,
    eos_token_id=1,
    bos_token_id=2,
    tie_word_embeddings=True,
    rope_theta=10000.0,
    attention_bias=False,
    attention_dropout=0.0,
    query_pre_attn_scalar=224,
    sliding_window=4096,
    final_logit_softcapping=30.0,
    attn_logit_softcapping=50.0,
    cache_implementation="hybrid",
    **kwargs,
):
    assert input_len == 1
    mb = ModelBuilder(f'Gemma2')
    input_shape = (1, hidden_size, input_len, 1)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    rope_weight = mb.RoPEWeight(dim=head_dim, input_pos=input_pos, input_len=input_len, base=rope_theta)

    for i in range(num_hidden_layers):
        x = DecoderBlock(mb, x, rope_weight, input_pos, input_len, num_attention_heads, num_key_value_heads, head_dim, hidden_size, intermediate_size, max_seq_len, block_idx=f'{i}')

    x = RMSNorm(mb, x, hidden_size)

    return mb


def GemmaLayer(
    input_len=1,
    input_pos=128,
    hidden_size=2304,
    intermediate_size=9216,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=256,
    max_seq_len=8192,
):
    mb = ModelBuilder(f'Gemma_layer')

    input_shape = (1, hidden_size, input_len, 1)
    x = mb.set_input_tensor(tensor_shape=input_shape)
    rope_weight = mb.RoPEWeight(dim=head_dim, input_pos=input_pos, input_len=input_len)
    x = DecoderBlock(mb, x, rope_weight, input_pos, input_len, num_attention_heads, num_key_value_heads, head_dim, hidden_size, intermediate_size, max_seq_len)

    return mb


def LayerNorm(mb, x, shape, name_postfix='', npu_support=False):
    if npu_support:
        return mb.LayerNorm(x, shape)
    if mb.model_dict[x].generic.is_first_tensor:
        return x
    out = mb.Concat([x], name='LayerNorm'+name_postfix)     # Baseline: Offload to the host
    out_tensor = mb.model_dict[out].output
    out_tensor = nn.LayerNorm(out_tensor.shape)(out_tensor) # XXX: Temporal solution just to compile
    mb.model_dict[out].output = out_tensor
    mb.model_dict[out].generic.output_tensor = out_tensor.detach().numpy()
    return out


def RMSNorm(mb, x, shape, name_postfix='', npu_support=False):
    name = 'RMSNorm' + name_postfix
    def RMSNormOp(tensor):
        input = tensor.permute(0, 3, 2, 1)  # batch, 1, seq_len, dim
        out = nn.RMSNorm(shape)(input)
        return out.permute(0, 3, 2, 1)
    return mb.HostProcessNode(x, RMSNormOp, name=name, mapping='cpu')


def Softmax(mb, x, name_postfix=''):
    name = 'Softmax' + name_postfix
    return mb.HostProcessNode(x, nn.Softmax(1), name=name, mapping='cpu')


def DecoderBlock(
    mb,
    input,
    rope_weight,
    input_pos,
    input_len,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=256,
    hidden_size=3072,
    intermediate_size=24576,
    max_seq_len=8192,
    cache_start_pos=0,
    block_idx=''
):
    print(f"hidden_size:{hidden_size}, int_size:{intermediate_size}")
    x = RMSNorm(mb, input, hidden_size, name_postfix=f'_D{block_idx}_pre_attn', npu_support=layernorm_support)
    x = MultiheadAttention(mb, x, rope_weight, input_pos, input_len, num_attention_heads, num_key_value_heads, head_dim, hidden_size, max_seq_len, cache_start_pos=cache_start_pos, block_idx=block_idx)
    x = RMSNorm(mb, x, hidden_size, name_postfix=f'_D{block_idx}_post_attn', npu_support=layernorm_support)
    x = mb.Sum(x, input, name=f'D{block_idx}_residual_conn1')
    y = x
    x = RMSNorm(mb, x, hidden_size, name_postfix=f'_D{block_idx}_pre_mlp', npu_support=layernorm_support)
    x = MLP(mb, x, input_len, hidden_size, intermediate_size, block_idx=block_idx)
    x = RMSNorm(mb, x, hidden_size, name_postfix=f'_D{block_idx}_post_mlp', npu_support=layernorm_support)
    return mb.Sum(y, x, name=f'D{block_idx}_residual_conn2')


def MultiheadAttention(mb, input, rope_weight, input_pos, input_len, num_attention_heads=16, num_key_value_heads=16, head_dim=256, hidden_size=3072, max_seq_len=8192, cache_start_pos=0, block_idx=''):
    assert num_attention_heads % num_key_value_heads == 0   # GQA
    query_per_kv = num_attention_heads // num_key_value_heads
    #input = {i: mb.DummyNode(input, name=f'D{block_idx}_H{i}_in') for i in range(0, num_attention_heads, query_per_kv)}
    out_w = {i: mb.Constant(tensor_shape=(1, head_dim, hidden_size, 1), name=f'D{block_idx}_H{i}_out_w') for i in range(0, num_attention_heads)}
    total_len = min(max_seq_len-cache_start_pos, input_pos+input_len-cache_start_pos)
    V_w = {i: mb.set_input_tensor(tensor_shape=(1, hidden_size, head_dim, 1), name=f'D{block_idx}_V_w{i}') for i in range(0, num_attention_heads)}
    V_T = {i: mb.Constant(tensor_shape=(1, total_len, head_dim, 1), name=f'D{block_idx}_V{i}_Cached') for i in range(0, num_key_value_heads)}
    mha_out = []
    prev_mha = None
    for i in range(0, num_attention_heads, query_per_kv):
        input_ = {k: mb.DummyNode(input, name=f'D{block_idx}_H{k}_in') for k in range(i, i+8)}
        Q = []
        for j in range(i, i + query_per_kv):
            Q.append(mb.RoPE(mb.MatMul([input_[j]], input_len, hidden_size, head_dim, name=f'D{block_idx}_Q{j}'), rope_weight, name=f'D{block_idx}_Q{j}_RoPE'))
        K = mb.MatMul([input_[i]], input_len, hidden_size, head_dim, name=f'D{block_idx}_K{i}')
        K = mb.RoPE(K, rope_weight, name=f'D{block_idx}_K{i}_RoPE')
        KCache = mb.CacheWrite(K, max_shape=(1, head_dim, max_seq_len, 1), write_offset=(0, input_pos, 0), write_shape=(head_dim, input_len, 1), name=f'D{block_idx}_K{i}_Cache')
        K = mb.CacheRead(KCache, read_offset=(0, cache_start_pos, 0), read_shape=(head_dim, total_len, 1), name=f'D{block_idx}_K{i}_Cached')
        V = mb.MatMul([input_[i]], input_len, hidden_size, head_dim, name=f'D{block_idx}_V{i}')        
        #V = mb.MatMul_binary_transpose(V_w[i], input_[i], head_dim, hidden_size, input_len, name=f'D{block_idx}_V_T{i}') 
        #VCache = mb.CacheWrite(V, max_shape=(1, max_seq_len, head_dim, 1), write_offset=(0, input_pos, 0), write_shape=(input_len, head_dim, 1), name=f'D{block_idx}_V{i}_Cache', write=False)
        #V_T = mb.CacheRead(VCache, read_offset=(0, cache_start_pos, 0), read_shape=(total_len, head_dim, 1), name=f'D{block_idx}_V{i}_Cached')
        for j in range(i, i + query_per_kv):
            QK_T = mb.MatMul_binary_transpose(Q[j-i], K, input_len, head_dim, total_len, name=f'D{block_idx}_QK_T{j}')
            A = Softmax(mb, QK_T, name_postfix=f'_D{block_idx}_H{j}')
            #mha_out.append(mb.MatMul_binary_transpose(QK_T, V, input_len, input_len, head_dim, name=(f'D{block_idx}_S_V{j}')))
            #S_V = mb.MatMul_binary_transpose(A, V, input_len, input_len, head_dim, name=(f'D{block_idx}_S_V{j}'))
            #S_V = mb.MatMul_binary_transpose(A, V_T[i//2], input_len, total_len, head_dim, name=(f'D{block_idx}_S_V{j}'))
            #V_T_out = mb.MatMul_binary_transpose(out_w[j], V, hidden_size, head_dim, total_len, name=f'D{block_idx}_WoV_T{j}')
            #mha_out = mb.MatMul_binary_transpose(A, V_T_out, total_len, total_len, hidden_size, name=f'D{block_idx}_H{j}_out')
            mha_out.append(mb.MatMul_binary_transpose(A, V_T[i//2], input_len, total_len, head_dim, name=(f'D{block_idx}_S_V{j}')))
            #if prev_mha is None:
            #    prev_mha = mha_out
            #else:
            #    prev_mha = mb.Sum(mha_out, prev_mha)
    concat = mb.Concat(mha_out, axis='c')
    out = mb.MatMul([concat], input_len, head_dim * num_attention_heads, hidden_size, name=f'D{block_idx}_MHA_out')
    return out
    #return prev_mha


def MLP(mb, input, input_len, hidden_size, intermediate_size, block_idx=''):
    x = mb.MatMul([input], input_len, hidden_size, intermediate_size, activation='GELU', name=f'D{block_idx}_gate_proj')
    y = mb.MatMul([input], input_len, hidden_size, intermediate_size, name=f'D{block_idx}_up_proj')
    x = mb.Mul(x, y)
    x = mb.MatMul([x], input_len, intermediate_size, hidden_size, name=f'D{block_idx}_down_proj')
    return x


def Gemma2_2b(
    input_len=512,
    input_pos=0,
    hidden_size=2304,
    intermediate_size=9216,
    num_hidden_layers=26,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=256,
    max_seq_len=8192,
):
    return Gemma2(input_len=input_len, input_pos=input_pos, hidden_size=hidden_size, intermediate_size=intermediate_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, head_dim=head_dim, max_seq_len=max_seq_len)


def offload_layers(mb: ModelBuilder, offload_name_list, mapping='cpu'):
    for name in offload_name_list:
        # Replace the previous generic_op instance with a HostProcessOp instance
        previous_generic_op = mb.model_dict[name].generic
        new_generic_op = HostProcessOp(name=name, input_layers=previous_generic_op.input_layers, output_tensor=previous_generic_op.output_tensor, mapping=mapping)
        mb.model_dict[name].generic = new_generic_op
    return mb


def GemmaLayer_pim_offload(
    input_len=1,
    input_pos=128,
    hidden_size=2304,
    intermediate_size=9216,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=256,
    max_seq_len=8192,
):
    scenario_id=0
    target_model = ""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "../../../hsim/configs/system.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        workload = config.get("workload", {})
        target_model = workload.get("model")
        scenario_id = workload.get("scenario_id")
        input_pos = workload.get("input_sequence_length")

    if (target_model == "gemma2-9B"):
        hidden_size = 3584
        intermediate_size = 14336
        num_attention_heads = 16
        num_key_value_heads = 8
        input_pos = 0

    mb = GemmaLayer(input_len=input_len, input_pos=input_pos, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, head_dim=head_dim, max_seq_len=max_seq_len)
    cpu_offload_name_list = ['D_residual_conn1', 'D_residual_conn2']
    
    if scenario_id == 0:
        pim_offload_name_list = []

    elif scenario_id == 1:
        pim_offload_name_list = [
            'D_K0', 'D_Q0', 'D_Q1', 'D_V0',
            'D_K2', 'D_Q2', 'D_Q3', 'D_V2',
            'D_K4', 'D_Q4', 'D_Q5', 'D_V4',
            'D_K6', 'D_Q6', 'D_Q7', 'D_V6'
        ]
    elif scenario_id == 2:
        pim_offload_name_list = [
            'D_K0', 'D_Q0', 'D_Q1',
            'D_K2', 'D_Q2', 'D_Q3',
            'D_K4', 'D_Q4', 'D_Q5',
            'D_K6', 'D_Q6', 'D_Q7',
        ]
    elif scenario_id == 3:
        pim_offload_name_list = [
            'D_K0', 
            'D_K2', 'D_Q2', 'D_Q3',
            'D_K4', 'D_Q4', 'D_Q5',
            'D_K6', 'D_Q6', 'D_Q7'
        ]
    elif scenario_id == 4:
        pim_offload_name_list = [
            'D_K0',
            'D_K2', 'D_Q2', 'D_Q3',
            'D_K4', 'D_Q4', 'D_Q5',
            'D_K6', 'D_Q6', 'D_Q7',
            'D_MHA_out'
        ]
    elif scenario_id == 5:
        pim_offload_name_list = [
            'D_K0',
            'D_K2', 'D_Q2', 'D_Q3',
            'D_K4', 'D_Q4', 'D_Q5',
            'D_K6', 'D_Q6', 'D_Q7',
            'D_MHA_out',
            'D_up_proj', 'D_gate_proj', 'Mul1', 'D_down_proj'
        ]
    # Partitioned LPDDR4 Best mapping
    elif scenario_id == 6:
        pim_offload_name_list = [
            'D_K0',
            'D_K2', 'D_Q2', 'D_Q3',
            'D_K4', 'D_Q4', 'D_Q5',
            'D_K6', 'D_Q6', 'D_Q7',
            'D_K0_RoPE', 'D_Q0_RoPE', 'D_Q1_RoPE', 
            'D_QK_T0', 'D_QK_T1',
            'D_MHA_out',
            'D_up_proj', 'D_gate_proj', 'Mul1', 'D_down_proj'
        ]
    # Unified Best mapping
    elif scenario_id == 7:
        pim_offload_name_list = [
            'D_K0', 'D_Q0', 'D_Q1', 'D_V0',
            'D_K2', 'D_Q2', 'D_Q3', 'D_V2',
            'D_K4', 'D_Q4', 'D_Q5', 'D_V4',
            'D_K6', 'D_Q6', 'D_Q7', 'D_V6',
            'D_MHA_out',
            'D_up_proj', 'D_gate_proj', 'Mul1', 'D_down_proj'
        ]
    # Partitioned DDR4 Best mapping (single-channel)
    elif scenario_id == 8:
        pim_offload_name_list = [
            'D_K0',
            'D_K2', 'D_Q2', 'D_Q3',
            'D_K4', 'D_Q4', 'D_Q5',
            'D_K6', 'D_Q6', 'D_Q7',
            'D_K0_RoPE', 'D_Q0_RoPE',
            'D_QK_T0',
            'D_MHA_out',
            'D_up_proj', 'D_gate_proj', 'Mul1', 'D_down_proj'
        ]
    # Partitioned DDR4 Best mapping (multi-channel)
    elif scenario_id == 9:
        pim_offload_name_list = [
            'D_K0',
            'D_K2', 'D_Q2', 'D_Q3',
            'D_K4', 'D_Q4', 'D_Q5',
            'D_K6', 'D_Q6', 'D_Q7',
            'D_K0_RoPE', 'D_Q0_RoPE',
            'D_QK_T0',
            'D_MHA_out',
            'D_up_proj', 'Mul1', 'D_down_proj'
        ]
    elif scenario_id == 10:
        pim_offload_name_list = [
            'D_K0',
            'D_K2',
            'D_K4',
            'D_K6', 'D_Q6', 'D_Q7',
            'D_K8', 'D_Q8', 'D_Q9',
            'D_K10', 'D_Q10', 'D_Q11',
            'D_K12', 'D_Q12', 'D_Q13',
            'D_K14', 'D_Q14', 'D_Q15',
            'D_K0_RoPE', 'D_Q0_RoPE', 'D_Q1_RoPE',
            'D_K2_RoPE', 'D_Q2_RoPE', 'D_Q3_RoPE',
            'D_K4_RoPE', 'D_Q4_RoPE', 'D_Q5_RoPE',
            'D_K6_RoPE', 'D_Q6_RoPE', 'D_Q7_RoPE',
            'D_QK_T0', 'D_QK_T1',
            'D_QK_T2', 'D_QK_T3',
            'D_QK_T4', 'D_QK_T5',
            'D_QK_T6', 'D_QK_T7',
            'D_S_V0', 'D_S_V1',
            'D_S_V2', 'D_S_V3',
            'D_S_V4', 'D_S_V5',
            'D_S_V6', 'D_S_V7',
            'D_MHA_out',
            'D_up_proj',
            #'D_gate_proj',
            'Mul1', 'D_down_proj'
        ]
    elif scenario_id == 11:
        pim_offload_name_list = [
            'D_K0', 'D_Q0', 'D_Q1', 'D_V0',
            'D_K2', 'D_Q2', 'D_Q3', 'D_V2',
            'D_K4', 'D_Q4', 'D_Q5', 'D_V4',
            'D_K6', 'D_Q6', 'D_Q7', 'D_V6',
            'D_K8', 'D_Q8', 'D_Q9', 'D_V8',
            'D_K10', 'D_Q10', 'D_Q11', 'D_V10',
            'D_K12', 'D_Q12', 'D_Q13', 'D_V12',
            'D_K14', 'D_Q14', 'D_Q15', 'D_V14',
            'D_MHA_out',
            'D_up_proj',
            'D_gate_proj','Mul1', 'D_down_proj'
        ] 
    mb = offload_layers(mb, cpu_offload_name_list, mapping='cpu')
    
    return offload_layers(mb, pim_offload_name_list, mapping='pim')

def Gemma2_2b_pim_offload(
    input_len=1,
    input_pos=0,
    hidden_size=2304,
    intermediate_size=9216,
    num_hidden_layers=26,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=256,
    max_seq_len=8192,
):
    mb = Gemma2_2b(input_len=input_len, input_pos=input_pos, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, head_dim=head_dim, max_seq_len=max_seq_len, num_hidden_layers=num_hidden_layers)
    cpu_offload_name_list = sum(([f'D{i}_residual_conn1', f'D{i}_residual_conn2'] for i in range(num_hidden_layers)), start=[])
    pim_offload_name_list = sum(([f'D{i}_Q6', f'D{i}_Q7', f'D{i}_K6', f'D{i}_Q6_RoPE', f'D{i}_Q7_RoPE', f'D{i}_K6_RoPE', f'D{i}_QK_T6', f'D{i}_QK_T7', f'D{i}_up_proj'] for i in range(num_hidden_layers)), start=[])
    mb = offload_layers(mb, cpu_offload_name_list, mapping='cpu')
    return offload_layers(mb, pim_offload_name_list, mapping='pim')

