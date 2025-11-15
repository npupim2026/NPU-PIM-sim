#include "partitioned.h"

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_partitioned(TraceGenerator& gen)
{
    const std::vector<std::string> pim_linear_cmds = { "mode_change1", "bank_to_grf", "mac", "grf_to_bank", "mode_change2" };
    const std::vector<std::string> pim_rope_cmds =   { "bank_to_grf1", "bank_to_grf2", "mul", "grf_to_bank" };
    const std::vector<std::string> pim_mul_cmds =    { "bank_to_grf1", "mul", "grf_to_bank" };

    /* Write RoPE weight */ gen.add_memory_trace(TraceType::WRITE, "RoPE_", HOST, {DRAM,PIM});
    /* RMSNorm_pre_attn  */ gen.add_host_trace("RMSNorm_D_pre_attn", {}, {DRAM,PIM});
    /* Start MIDAP       */ gen.add_trace(TraceType::COMPUTE, "", NPU);
    
    gen.add_gemv_trace("D_K0", pim_linear_cmds, PIM);
    gen.add_gemv_trace("D_K2", pim_linear_cmds, DRAM);
    gen.add_gemv_trace("D_Q2", pim_linear_cmds, DRAM);
    gen.add_gemv_trace("D_Q3", pim_linear_cmds, DRAM);
    gen.add_gemv_trace("D_K4", pim_linear_cmds, DRAM);
    gen.add_gemv_trace("D_Q4", pim_linear_cmds, DRAM);
    gen.add_gemv_trace("D_Q5", pim_linear_cmds, DRAM);
    gen.add_gemv_trace("D_K6", pim_linear_cmds, DRAM);
    gen.add_gemv_trace("D_Q6", pim_linear_cmds, DRAM);
    gen.add_gemv_trace("D_Q7", pim_linear_cmds, DRAM);
    
    gen.add_rope_trace("D_K0_RoPE", pim_rope_cmds);
    gen.add_rope_trace("D_Q0_RoPE", pim_rope_cmds);
    gen.add_gemv_trace("D_QK_T0", pim_linear_cmds, PIM);
    gen.add_host_trace("Softmax_D_H0", {PIM}, {DRAM});
    gen.add_host_trace("Softmax_D_H1", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H3", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H2", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H5", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H4", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H7", {DRAM}, {DRAM});
    gen.add_host_trace("Softmax_D_H6", {DRAM}, {DRAM});

    /* MHA_out           */ gen.add_gemv_trace("D_MHA_out", pim_linear_cmds, PIM);
    /* RMSNorm_post_attn */ gen.add_host_trace("RMSNorm_D_post_attn", {PIM}, {DRAM});
    /* residual_conn1    */ gen.add_host_trace("D_residual_conn1", {DRAM}, {DRAM});
    /* RMSNorm_pre_mlp   */ gen.add_host_trace("RMSNorm_D_pre_mlp", {DRAM}, {DRAM,PIM});

    /* MLP */
    /* up_proj   */ gen.add_gemv_trace("D_up_proj", pim_linear_cmds, PIM);
    /* mul1      */ gen.add_rope_trace("Mul1", pim_mul_cmds);
    /* down_proj */ gen.add_gemv_trace("D_down_proj", pim_linear_cmds, PIM);

    /* RMSNorm_post_mlp */ gen.add_host_trace("RMSNorm_D_post_mlp", {PIM}, {DRAM});
    /* residual_conn2   */ gen.add_host_trace("D_residual_conn2", {DRAM}, {});

    /* Terminate */ gen.add_trace(TraceType::TERMINATE, "", HOST);

    return gen.trace_queue;
}
