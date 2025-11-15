#include "host.h"

bool is_pim_busy = false;
sc_event pim_compute_done_ev;

Host::Host(sc_module_name name, Logger* logger)
: master("master"), clock("clock"), peq(this, &Host::peq_cb), logger(logger)
{
    SC_THREAD(process_trace);
    master.register_nb_transport_bw(this, &Host::nb_transport_bw);

    SC_METHOD(clock_negedge);
    sensitive << clock.neg();
    dont_initialize();

    mem_req_size = cfgs.get_dram_req_size();
    pim_wait_map = cfgs.get_pim_wait_map();
    pim_sync_map = cfgs.get_pim_sync_map();
    host_profile = cfgs.get_host_profile();
    pim_profile = cfgs.get_pim_profile();
    layer_tile_map = cfgs.get_layer_tile_map();
    
    trace_queue = trace_generator.generate_trace();
};

void Host::print_log(const tlm_generic_payload* trans)
{
    const std::string& layer = trans->get_layer();
    uint32_t burst_id = trans->get_bst_id();
    uint32_t burst_count = trans->get_bst_size();

    if (trans->get_command() == TLM_READ_COMMAND) {
        if (trans->get_dst_id() == DRAM ) {
            std::cout << "[Host->DRAM][READ] " << layer << ", " << burst_id+1 << "/" << burst_count << ", cycle: " << current_cycle() << "\n";
        } else if (trans->get_dst_id() == PIM) {
            if (trans->get_pim_cmd() == "addertree") {
                std::cout << "[Host->PIM][AdderTree][READ] " << layer << ", " << burst_id+1 << "/" << burst_count << "\n";
            } else {
                std::cout << "[Host->PIM][READ] " << layer << ", " << burst_id+1 << "/" << burst_count << "\n";
            }
        }
    } else if (trans->get_command() == TLM_WRITE_COMMAND) {
        if (trans->get_dst_id() == DRAM ) {
            std::cout << "[Host->DRAM][WRITE] " << layer << ", " << burst_id+1 << "/" << burst_count << ", cycle: " << current_cycle() << "\n";
        } else if (trans->get_dst_id() == PIM) {
            if (trans->get_pim_cmd() == "addertree") {
                std::cout << "[Host->PIM][AdderTree][WRITE] " << layer << ", " << burst_id+1 << "/" << burst_count << "\n";
            } else {
                std::cout << "[Host->PIM][WRITE] " << layer << ", " << burst_id+1 << "/" << burst_count << "\n";
            }
        }
    } else {
        std::cout << "[Host->PIM][COMPUTE] " << layer << " " << trans->get_pim_cmd() << ", " << burst_id+1 << "/" << burst_count << "\n";
    }
}

void Host::clock_negedge()
{
    /* No pending transactions */
    if (pending_queue.empty()) {
        return;
    }
    
    /* Dequeue next transaction */
    tlm_generic_payload* trans = pending_queue.front();
    pending_queue.pop_front();

    /* Select target socket (0 = NPU, 1 = Arbiter) */
    int id = (trans->get_dst_id() == NPU) ? 0 : 1;

    switch (trans->get_dst_id()) {
        case NPU:
            id = 0;
            break;
        case DRAM:
        case PIM:
            id = 1;
            break;
        case MCU:
            id = 2;
            break;
    }

    if (id == 1) { print_log(trans); }

    tlm_phase phase = BEGIN_REQ;
    tlm_sync_enum reply = master[id]->nb_transport_fw(*trans, phase, t);
    assert(reply == TLM_UPDATED);

    active_cycle++;
}

void Host::process_trace()
{
    while (true) {
        if (!trace_queue.empty()) {
            auto trace = trace_queue.front();
            trace_queue.pop_front();
            switch (trace->type) {
                case TraceType::WRITE:
                case TraceType::READ:
                    handle_memory_trace(std::dynamic_pointer_cast<TraceGenerator::MemoryTrace>(trace));
                    break;
                case TraceType::COMPUTE:
                    handle_compute_trace(std::dynamic_pointer_cast<TraceGenerator::ComputeTrace>(trace));
                    break;
                case TraceType::PIM:
                    handle_pim_trace(std::dynamic_pointer_cast<TraceGenerator::PimTrace>(trace));
                    break;
                case TraceType::TERMINATE:
                    active_host = 0;
                    break;
            }
        }
        if (active_dram == 0) break;
        wait(1, SC_NS);
    }
    logger->print_info(""); /* Print log */
    logger->print_info(cfgs.get_cycle_log_file()); /* Save into file */
    sc_stop();
}

void Host::handle_memory_trace(const std::shared_ptr<TraceGenerator::MemoryTrace>& trace)
{
    const std::string& layer = trace->layer;
    /* Handle dependencies */
    if (trace->type == TraceType::READ) {
        /* Ensure data is ready before reading */
        wait_for_sync(trace->id, trace->dst);
    }
    else if (trace->type == TraceType::WRITE) {
        /* Wait for compute completion before writing */
        if (!compute_done_map[layer] && (layer.find("RoPE_") == std::string::npos)) {
            wait_for_compute(layer);
        }
    }
    logger->update_start(layer, HOST);

    /* Split request into multiple transactions */
    unsigned int trans_num = (trace->size + mem_req_size - 1) / mem_req_size;

    for (size_t i = 0; i < trans_num; i++) {
        tlm_generic_payload* trans = generate_trans(
            trace->type,
            trace->src,
            trace->dst,
            trace->address + (i*mem_req_size),
            i,
            trans_num,
            layer
        );
        trans->set_head(i == 0);

        if (trace->type == TraceType::WRITE) {
            trans->set_last(i == (trans_num - 1));
        }

        pending_queue.push_back(trans);
    }

    /* Wait until all read responses are received */
    if (trace->type == TraceType::READ && trace->src == HOST) {
        wait_for_responses(trans_num, layer, false);
    }
}

void Host::handle_compute_trace(const std::shared_ptr<TraceGenerator::ComputeTrace>& trace)
{
    const std::string& layer = trace->layer;
    if (trace->device == HOST) {
        logger->update_start(layer, HOST);
        std::cout << layer << " compute start! " << current_cycle() << std::endl;
        /* Simulate computation delay */
        const auto op_type = get_op_type(layer);
        unsigned int compute_delay = host_profile[op_type];
        std::cout << "[Host][COMPUTE] " << layer << " (" << compute_delay << " ns)" << std::endl;
        wait(compute_delay, SC_NS);

        /* Compute completion */
        host_compute_done = true;
        compute_done_map[layer] = true;
        host_compute_done_ev.notify();

        std::cout << layer << " compute end! " << current_cycle() << std::endl;
        logger->update_end(layer, HOST);
    }
    else if (trace->device == NPU || trace->device == MCU) {
        /* NPU: Init compute, MCU: Macro PIM command */
        tlm_generic_payload* trans = m_mm.allocate();
        trans->acquire();
        trans->set_command(TLM_COMPUTE_COMMAND);
        trans->set_dst_id(trace->device);
        trans->set_layer(layer);
        pending_queue.push_back(trans);
    }
    else {
        SC_REPORT_FATAL("Host", "Unknown device type in ComputeTrace");
    }
}

void Host::process_addertree(const std::string& layer, const std::string& cmd, unsigned int next)
{
    /* Wait for PIM computation end */
    auto it = pim_sync_map.find(layer);
    if (it != pim_sync_map.end()) {
        unsigned int sync_id = std::get<0>(it->second);
        wait_for_sync(sync_id, PIM);
    }    

    auto [sync_id, size, address] = it->second;
    unsigned int read_trans_num = (size*16)/mem_req_size; // 16 FP16 elements per dim

    /* Read input from PIM */
    for (int i = 0; i < read_trans_num; i++) {
        tlm_generic_payload *read_trans = generate_trans(TraceType::READ, HOST, PIM, 0, i, read_trans_num, layer, cmd);
        pending_queue.push_back(read_trans);
    }

    /* Wait until all read responses are received */
    wait_for_responses(read_trans_num, layer, true);

    /* Simulate AdderTree computation delay */
    unsigned int delay = host_profile["addertree"];
    std::cout << "[Host][AdderTree][COMPUTE] " << layer << "\n";
    wait(delay, SC_NS);

    /* Write result back to PIM */
    unsigned int write_trans_num = (size + mem_req_size - 1) / mem_req_size;
    for (int i = 0; i < write_trans_num; i++) {
        tlm_generic_payload *write_trans = generate_trans(TraceType::WRITE, HOST, next, 0, i, write_trans_num, layer, cmd);
        write_trans->set_last(i == write_trans_num - 1);
        pending_queue.push_back(write_trans);
    }
}

void Host::handle_pim_trace(const std::shared_ptr<TraceGenerator::PimTrace>& trace)
{
    const std::string& layer = trace->layer;
    const std::string& micro_cmd = trace->cmd;
    unsigned int next_memory = trace->next;

    if (micro_cmd == "addertree") {
        process_addertree(layer, micro_cmd, next_memory);
        return;
    }

    /* Handle PIM sub-ops */
    const auto& profile = pim_profile[layer];
    unsigned int delay = profile.cmd_map.at(micro_cmd).delay;
    unsigned int count = profile.cmd_map.at(micro_cmd).count;
    size_t num_cmds = profile.cmd_map.size();
    unsigned int tile_num = layer_tile_map[layer];
    cmd_id++;

	delay = delay * 0.33;

    /* Wait for input dependencies to be ready */
    auto range = pim_wait_map.equal_range(layer);
    for (auto it = range.first; it != range.second; ++it) {
        unsigned int id = std::get<0>(it->second);
        if (!pim_sync_obj.check_signal(id)) {
            wait(*pim_sync_obj.get_event(id));
        }
    }

    if (layer.find("RoPE") != std::string::npos) {
        if (layer != previous_layer) {
            if (is_pim_busy) {
                wait(pim_compute_done_ev);  /* PIM이 이전 RoPE 연산을 마칠 때까지 기다림 */
            }
            is_pim_busy = true;
        }
        previous_layer = layer;
    }

    logger->update_start(layer, HOST);

    /* Generate micro PIM transactions */
    for (int i = 0; i < count; i++) {
        tlm_generic_payload *pim_trans = generate_trans(TraceType::PIM, HOST, PIM, delay, i, count, layer, micro_cmd);

        pim_trans->set_last((cmd_id == num_cmds*tile_num) && (i == count - 1));
        if (pim_trans->is_last()) { cmd_id = 0; }
        pending_queue.push_back(pim_trans);
    }
}

tlm_generic_payload* Host::generate_trans(TraceType type, unsigned int src, unsigned int dst, unsigned int addr,
                                        unsigned int burst_id, unsigned int burst_size,
                                        const std::string& layer, const std::string& micro_cmd)
{
    tlm_generic_payload* trans = m_mm.allocate();
    trans->acquire();
    trans->set_pim_cmd(micro_cmd);
    trans->set_address(addr);
    trans->set_bst_size(burst_size);
    trans->set_bst_id(burst_id);
    trans->set_src_id(src);
    trans->set_dst_id(dst);
    trans->set_layer(layer);
    trans->set_command((type == TraceType::PIM) ? TLM_PIM_COMMAND :
        (type == TraceType::WRITE) ? TLM_WRITE_COMMAND : TLM_READ_COMMAND);
    return trans;
}

void Host::wait_for_sync(unsigned int id, unsigned int dst)
{
    SyncObject& sync_obj = (dst == DRAM) ? dram_sync_obj : pim_sync_obj;
    if (!sync_obj.check_signal(id)) {
        wait(*sync_obj.get_event(id));
    }
}

void Host::wait_for_compute(const std::string& layer)
{
    if (!host_compute_done) {
        wait(host_compute_done_ev);
    }
    host_compute_done = false;
}

void Host::wait_for_responses(unsigned int trans_num, const std::string& layer, bool addertree)
{
    unsigned int response_received = 0;

    while (response_received < trans_num) {
        while (!read_response_queue.empty()) {
            read_response_queue.pop_front();
            response_received++;
            if (addertree) {
                std::cout << "[Host][AdderTree][READ RESPONSE] " << layer << ", " << response_received << "/" << trans_num << std::endl;
            } else {
                std::cout << "[Host][READ RESPONSE] " << layer << ", " << response_received << "/" << trans_num << std::endl;
            }
        }

        if (response_received < trans_num) {
            wait(read_response_ev);
        }
    }
}

tlm_sync_enum Host::nb_transport_bw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
    peq.notify(trans, phase, SC_ZERO_TIME);
	return TLM_UPDATED;
}

void Host::peq_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
    read_response_queue.push_back(&trans);
    read_response_ev.notify();
}

const std::string& Host::get_op_type(const std::string& layer)
{
    static const std::string SOFTMAX = "softmax";
    static const std::string RMSNORM = "rmsnorm";
    static const std::string CONN = "residual";
    static const std::string GELU = "gelu";
    if (layer.find("Softmax") != std::string::npos) return SOFTMAX;
    else if (layer.find("RMSNorm") != std::string::npos) return RMSNORM;
    else if (layer == "gelu") return GELU;
    return CONN;
}
