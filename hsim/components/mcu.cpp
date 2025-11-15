#include "mcu.h"

Mcu::Mcu(sc_module_name name, Logger* logger)
: master("master"), slave("slave"), clock("clock"), peq(this, &Mcu::peq_cb), logger(logger)
{
	SC_THREAD(process_trace);
    master.register_nb_transport_bw(this, &Mcu::nb_transport_bw);
	slave.register_nb_transport_fw(this, &Mcu::nb_transport_fw);
	
    SC_METHOD(clock_negedge);
    sensitive << clock.neg();
    dont_initialize();

    mem_req_size = cfgs.get_dram_req_size();
    host_profile = cfgs.get_host_profile();
    pim_sync_map = cfgs.get_pim_sync_map();
    pim_wait_map = cfgs.get_pim_wait_map();
    pim_profile = cfgs.get_pim_profile();
    layer_tile_map = cfgs.get_layer_tile_map();    
}

Mcu::~Mcu() {}

void Mcu::process_trace()
{
    while (true) {
        if (trans_queue.empty()) {
            wait(trans_arrival_ev);
            continue;
        } 
        auto* trans = trans_queue.front();
        trans_queue.pop_front();
        
        const std::string& layer = trans->get_layer();

        /* Wait for input dependencies */
       wait_for_dependencies(layer);

        auto tile_it = layer_tile_map.find(layer);
        if (tile_it == layer_tile_map.end()) continue;

        const auto& profile = pim_profile[layer];
        const auto& cmd_list = profile.ordered_cmd_name;
        unsigned int tile_num = tile_it->second;

        logger->update_start(layer, MCU);

        generate_pim_trans(layer, profile, cmd_list, tile_num);

        if (layer.find("RoPE_")==std::string::npos && layer!="Mul1") {
            process_addertree(layer, "addertree");
        }
    }
}

void Mcu::process_addertree(const std::string& layer, const std::string& cmd)
{
    
    auto it = pim_sync_map.find(layer);
    if (it == pim_sync_map.end()) return;

    unsigned int sync_id;
    uint64_t size, address;
    std::tie(sync_id, size, address) = it->second;

    /* Wait for PIM computation end */
    wait_for_sync(sync_id);

    /* Read input from PIM */
    unsigned int read_trans_num = (size*16)/mem_req_size; // 16 FP16 elements per dim
    for (int i = 0; i < read_trans_num; i++) {
        auto* read_trans = generate_trans(TraceType::READ, MCU, cmd, layer, i, read_trans_num);
        fw_queue.push_back(read_trans);
    }

    /* Wait until all read responses are received */
    wait_for_responses(read_trans_num, layer, true);

    /* Simulate AdderTree computation delay */
    unsigned int delay = host_profile["addertree"];
    std::cout << "[MCU][AdderTree][COMPUTE] " << layer << ", Delay:" << delay << "ns\n";
    wait(delay, SC_NS);

    /* Write result back to PIM */
    unsigned int write_trans_num = (size + mem_req_size - 1) / mem_req_size;
    for (int i = 0; i < write_trans_num; i++) {
        auto* write_trans = generate_trans(TraceType::WRITE, MCU, cmd, layer, i, write_trans_num);
        write_trans->set_last(i == write_trans_num - 1);
        fw_queue.push_back(write_trans);
    }    
}

void Mcu::generate_pim_trans(const std::string& layer, const PIMCmdProfile& profile, const std::vector<std::string>& cmd_list,
                            unsigned int tile_num)
{
    for (int i = 0; i < tile_num; i++) {
        for (size_t cmd_idx = 0; cmd_idx < cmd_list.size(); ++cmd_idx) {
            const std::string& micro_cmd = cmd_list[cmd_idx];
            const auto& cmdinfo = profile.cmd_map.at(micro_cmd);
            
            /* Generate micro PIM transactions */
            for (int j = 0; j < cmdinfo.count; j++) {
                auto *pim_trans = generate_trans(TraceType::PIM, MCU, micro_cmd, layer, j, cmdinfo.count);
                pim_trans->set_address(cmdinfo.delay * 0.88);
                bool is_last = (i == (tile_num-1)) && (cmd_idx == (cmd_list.size()-1)) && (j == cmdinfo.count-1);
                pim_trans->set_last(is_last);
                fw_queue.push_back(pim_trans);
            }
        }
    }
}

void Mcu::wait_for_dependencies(const std::string& layer)
{
    auto range = pim_wait_map.equal_range(layer);
    for (auto it = range.first; it != range.second; ++it) {
        unsigned int id = std::get<0>(it->second);
        if (!pim_sync_obj.check_signal(id)) {
            wait(*pim_sync_obj.get_event(id));
        }
    }
}

void Mcu::wait_for_responses(unsigned int trans_num, const std::string& layer, bool addertree)
{
    unsigned int response_received = 0;

    while (response_received < trans_num) {
        while (!read_response_queue.empty()) {
            read_response_queue.pop_front();
            response_received++;
            if (addertree) {
                std::cout << "[MCU][AdderTree][READ RESPONSE] " << layer << ", " << response_received << "/" << trans_num << std::endl;
            } else {
                std::cout << "[MCU][READ RESPONSE] " << layer << ", " << response_received << "/" << trans_num << std::endl;
            }
        }

        if (response_received < trans_num) {
            wait(read_response_ev);
        }
    }    
}

void Mcu::clock_negedge()
{
    if (!fw_queue.empty()) {
        tlm_generic_payload* trans = fw_queue.front();
        fw_queue.pop_front();
        tlm_phase phase = BEGIN_REQ;
        tlm_sync_enum reply = master->nb_transport_fw(*trans, phase, t);
        assert(reply == TLM_UPDATED);

        print_log(trans);
    }
}

tlm_sync_enum Mcu::nb_transport_fw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
    peq.notify(trans, phase, SC_ZERO_TIME);
	return TLM_UPDATED;
}

void Mcu::peq_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
    trans_queue.push_back(&trans);
    trans_arrival_ev.notify();
}

tlm_sync_enum Mcu::nb_transport_bw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
    read_response_queue.push_back(&trans);
    read_response_ev.notify();
	return TLM_UPDATED;
}

tlm_generic_payload* Mcu::generate_trans(TraceType type, unsigned int src, const std::string& micro_cmd,
                                        const std::string& layer, unsigned int burst_id, unsigned int burst_size)
{
    auto* trans = m_mm.allocate();
    trans->acquire();
    trans->set_pim_cmd(micro_cmd);
    trans->set_src_id(src);
    trans->set_layer(layer);
    trans->set_bst_size(burst_size);
    trans->set_bst_id(burst_id);
    trans->set_command((type == TraceType::PIM) ? TLM_PIM_COMMAND :
        (type == TraceType::WRITE) ? TLM_WRITE_COMMAND : TLM_READ_COMMAND);
    return trans;
}

void Mcu::wait_for_sync(unsigned int id)
{
    if (!pim_sync_obj.check_signal(id)) {
        wait(*pim_sync_obj.get_event(id));
    }
}

void Mcu::print_log(const tlm_generic_payload* trans)
{
    const std::string& layer = trans->get_layer();
    uint32_t burst_id = trans->get_bst_id();
    uint32_t burst_count = trans->get_bst_size();

    if (trans->get_pim_cmd() == "addertree") {
        if (trans->get_command() == TLM_WRITE_COMMAND) {
            std::cout << "[MCU->PIM][AdderTree][WRITE] " << layer << ", " << burst_id+1 << "/" << burst_count << "\n";
        } else {
            std::cout << "[MCU->PIM][AdderTree][READ] " << layer << ", " << burst_id+1 << "/" << burst_count << "\n";
        }
    } else if (trans->get_command() == TLM_PIM_COMMAND) {
        std::cout << "[MCU->PIM][COMPUTE] " << trans->get_layer() << " " << trans->get_pim_cmd() << ", " << burst_id+1 << "/" << burst_count << "\n";
    }
}