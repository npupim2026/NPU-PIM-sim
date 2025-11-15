#include "baseline.h"

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_baseline(TraceGenerator& gen)
{
    /* Write RoPE weight */
    gen.add_memory_trace(TraceType::WRITE, "RoPE_", HOST, {DRAM});

    /* RMSNorm_pre_attn */
    gen.add_host_trace("RMSNorm_D_pre_attn", {}, {DRAM});

    /* Start MIDAP */
    gen.add_trace(TraceType::COMPUTE, "", NPU);

    /*---- Softmax ----*/
    for (size_t hd_id = 0; hd_id < cfgs.get_attn_head_num(); hd_id++) {
        gen.add_host_trace("Softmax_D_H" + std::to_string(hd_id), {DRAM}, {DRAM});
    }

    /* RMSNorm_post_attn */
    gen.add_host_trace("RMSNorm_D_post_attn", {DRAM}, {DRAM});

    /* residual_conn1 */
    gen.add_host_trace("D_residual_conn1", {DRAM}, {DRAM});

    /* RMSNorm_pre_mlp */
    gen.add_host_trace("RMSNorm_D_pre_mlp", {DRAM}, {DRAM});

   /* RMSNorm_post_mlp */
    gen.add_host_trace("RMSNorm_D_post_mlp", {DRAM}, {DRAM});

    /* residual_conn2 */
    gen.add_host_trace("D_residual_conn2", {DRAM}, {});

    /* Terminate */
    gen.add_trace(TraceType::TERMINATE, "Terminate", HOST);

    return gen.trace_queue;
}