#include "mem_wrapper.h"

MEMWrapper::MEMWrapper(sc_module_name name, Logger* logger)
: slave("slave"), clock("clock"), peq(this, &MEMWrapper::peq_cb), logger(logger)
{
	SC_THREAD(simulate_dram);

	SC_METHOD(clock_posedge);
    sensitive << clock.pos();
	dont_initialize();

	SC_METHOD(clock_negedge);
    sensitive << clock.neg();
	dont_initialize();
	
	slave.register_nb_transport_fw(this, &MEMWrapper::nb_transport_fw);

	dram_req_size = cfgs.get_dram_req_size();
    host_sync_map = cfgs.get_host_sync_map();
    pim_sync_map = cfgs.get_pim_sync_map();
	init_dram();
}

MEMWrapper::~MEMWrapper()
{
	if (ramulator2_bridge) {
		delete ramulator2_bridge;
	}
	if (dramsim3_bridge) {
		// dramsim3_bridge->print_stats();
		delete dramsim3_bridge;
	}	
}

void MEMWrapper::init_dram()
{
	std::string type = cfgs.get_dram_type();
	std::string preset = cfgs.get_dram_preset();
	std::string config = cfgs.get_dram_config();
	uint32_t channel_num = cfgs.get_dram_channels();

	if (cfgs.get_dram_backend() == "ramulator2") {
		std::string path(RAMULATOR_PATH);
		std::string cfg_path = path + "configs/" + type + "/" + config + "_" + preset
									+ "_ch" + std::to_string(channel_num) + ".yaml";
		ramulator2_bridge = new ramulator2::Bridge(cfg_path.c_str(), 0);
	} else {
		std::string path(DRAMSIM3_PATH);
		std::string cfg_path = path + "configs/" + type + "/" + config + "_" + preset
									+ "_ch" + std::to_string(channel_num) + ".ini";
		dramsim3_bridge = new dramsim3::Bridge(cfg_path.c_str(), path.c_str());
	}
}

void MEMWrapper::clock_posedge() {
    if (!fw_queue.empty()) {
		tlm_generic_payload *trans = fw_queue.front();
		fw_queue.pop_front();
		mem_request.push_back(trans);
    }
}

void MEMWrapper::clock_negedge() {
	if (!bw_queue.empty())  {
		tlm_generic_payload* trans = bw_queue.front();
		bw_queue.pop_front();

		bool is_read = (trans->get_command() == TLM_READ_COMMAND) ? true : false;
		backward_trans(trans, is_read);
		m_outstanding--;
	}
}

tlm_sync_enum MEMWrapper::nb_transport_fw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t) {
	peq.notify(trans, phase, SC_ZERO_TIME);
	return TLM_UPDATED;
}

void MEMWrapper::peq_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
	fw_queue.push_back(&trans);
}

unsigned int MEMWrapper::find_sync_id(std::string layer)
{
    auto it = host_sync_map.find(layer);
	if (it != host_sync_map.end()) {
		const auto& [id, size, address] = it->second;
		return id;
	} else {
		throw std::runtime_error("[DRAM][ERROR] Address not found in map.");
	}
}

void MEMWrapper::simulate_dram()
{
    double period = 1 / cfgs.get_dram_freq();
	unsigned int sync_id;
	int wack_num = 0;

	while (1) {
		/* Get completed request from the DRAM simulator */
		tlm_generic_payload* trans = nullptr;
        if (cfgs.get_dram_backend() == "ramulator2") {
            trans = ramulator2_bridge->getCompletedCommand();
        } else {
            trans = dramsim3_bridge->getCompletedCommand();
        }

		if (trans) {
			auto cmd = trans->get_command();
			auto src_id = trans->get_src_id();
			const std::string& layer = trans->get_layer();

			if (cmd == TLM_READ_COMMAND) {
				bw_queue.push_back(trans);
			} else if (cmd == TLM_WRITE_COMMAND) {
				if (src_id == HOST && trans->get_pim_cmd() == "addertree") {
					complete_write_request(trans);
					m_outstanding--;
				}				
				/* Signal when HOST write (burst) is done */
				else if (src_id == HOST) {
					wack_num++;
					if (wack_num == (trans->get_bst_size())) {
						wack_num = 0;
						unsigned int sync_id = find_sync_id(layer);
						std::cout << "[DRAM][SIGNAL] " << layer << " (ID:" << sync_id << ")\n";
						dram_sync_obj.signal(sync_id);

						logger->update_end(layer, HOST);
					}
					m_outstanding--;
					trans->release();
				}

				else if (src_id == NPU) {
					bw_queue.push_back(trans);
				}
			}
		}
		
		/* Send command to the DRAM simulator */
		while (!mem_request.empty()) {
			tlm_generic_payload *req_trans = mem_request.front();
			mem_request.pop_front();
			m_outstanding++;

            if (cfgs.get_dram_backend() == "ramulator2") {
                ramulator2_bridge->sendCommand(*req_trans);
            } else {
                dramsim3_bridge->sendCommand(*req_trans);
            }
		}

		/* Simulate the DRAM simulator */
        if (cfgs.get_dram_backend() == "ramulator2") {
            ramulator2_bridge->ClockTick();
        } else {
            dramsim3_bridge->ClockTick();
        }

		/* Check if DRAM simulation can stop */
		if (active_host == 0 && active_core == 0 && m_outstanding == 0) {
			active_dram = 0;
			break;
		}

		/* Wait for the next clock period */
		wait(period, SC_NS);
		total_cycle++;
    }
}

void MEMWrapper::backward_trans(tlm_generic_payload* trans, bool read) {
	tlm_phase phase = BEGIN_RESP;
	tlm_sync_enum reply = slave->nb_transport_bw(*trans, phase, t);
	assert(reply == TLM_UPDATED);
}

void MEMWrapper::complete_write_request(tlm::tlm_generic_payload* trans)
{	
	if (trans->is_last()) {
		const std::string& layer = trans->get_layer();
		BaseMap* base_map = (trans->get_pim_cmd() == "addertree") ? &pim_sync_map : &host_sync_map;	

		signal_sync(*base_map, layer);

		if (trans->get_src_id() == HOST) {
			logger->update_end(layer, HOST);
		}
		else if (trans->get_src_id() == MCU) {
			logger->update_end(layer, MCU);
		}
	}
	trans->release();
}

void MEMWrapper::signal_sync(BaseMap& map, const std::string& layer)
{
	auto it = map.find(layer);
	if (it != map.end()) {
		unsigned int sync_id = std::get<0>(it->second);
		std::cout << "[PIM][AdderTree][SIGNAL] " << layer << " (ID:" << sync_id << "), cycle:" << current_cycle() << "\n";
		if (layer == "D_V0" || layer == "D_V2" || layer == "D_V4" || layer == "D_V6" ||
			layer == "D_V8" || layer == "D_V10" || layer == "D_V12" || layer == "D_V14") {
			dram_sync_obj.signal(sync_id);
		}
		pim_sync_obj.signal(sync_id);
	} else {
		std::cerr << "[PIM][ERROR] Key[" << layer << "] not found in the map!\n";
	}
}