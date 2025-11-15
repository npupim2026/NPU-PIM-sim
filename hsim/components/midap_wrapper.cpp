#include "midap_wrapper.h"

MIDAPWrapper::MIDAPWrapper(sc_module_name name, Logger* logger)
: sc_module(name), name(name), master("master"), slave("slave"), clock("clock"), logger(logger),
peq_fw(this, &MIDAPWrapper::peq_fw_cb), peq_bw(this, &MIDAPWrapper::peq_bw_cb)
{
    SC_THREAD(periodic_process);

    SC_METHOD(clock_posedge);
    sensitive << clock.pos();
    dont_initialize();

    SC_METHOD(clock_negedge);
    sensitive << clock.neg();
    dont_initialize();

	master.register_nb_transport_bw(this, &MIDAPWrapper::nb_transport_bw);
	slave.register_nb_transport_fw(this, &MIDAPWrapper::nb_transport_fw);

	packet_size = cfgs.get_packet_size();
    mem_req_size = cfgs.get_dram_req_size();
	pim_sync_map = cfgs.get_pim_sync_map();
	pim_sync_id_map = cfgs.get_pim_sync_id_map();
	pim_wait_id_map = cfgs.get_pim_wait_id_map();
};

MIDAPWrapper::~MIDAPWrapper()
{
	delete communicator;
}

void MIDAPWrapper::periodic_process()
{
	Packet packet;
	int cur_cycle = 0, last_req_cycle = 0;

    while (true) {
		if (status == TERMINATED) {
            active_core = 0;
            break;
        }

		if (status == RUNNING) {
            while (true) {
                if (communicator->irecv_packet(&packet) != 0)
                    break;
				if (req_to_read) {
					wait(1, SC_NS);
				}
                else {
					wait(SC_ZERO_TIME);
				}
            }
			elapse_cycle(packet.cycle);
           	handle_packet(&packet);
        } 
		wait(1, SC_NS);
    }
}

void MIDAPWrapper::clock_posedge() {
	if (rack_incoming){
		rack_queue.push_back(rack_incoming);
		rack_incoming = nullptr;
	}
	if (wack_incoming){
		wack_queue.push_back(wack_incoming);
		wack_incoming = nullptr;
	}		
}

void MIDAPWrapper::clock_negedge() {
	/* READ */
    if (!r_queue.empty()) {
		tlm_generic_payload* trans = r_queue.front();
		r_queue.pop_front();
		forward_trans(trans);
	}
    
	/* WRITE */
    if (!w_queue.empty()) {
		tlm_generic_payload* trans = w_queue.front();
		w_queue.pop_front();
		forward_trans(trans);
	}

	/* Send RACK */
    if (!rack_queue.empty()) {
		tlm_generic_payload* trans = rack_queue.front();
		rack_queue.pop_front();	
		handle_rack(trans);
		outstanding--;
	}

	/* Send WACK */
   	if (!wack_queue.empty()) {
    	tlm_generic_payload* trans = wack_queue.front();
		wack_queue.pop_front();	
		handle_wack(trans);
		outstanding--;
	}
	
	if (!sync_queue.empty() && (outstanding == 0)) {
        unsigned int sync_id = sync_queue.front();
        sync_queue.pop_front();
		
		std::cout << "[MIDAP][SIGNAL] " << layer << ", ID:" << sync_id << ", " << current_cycle() << "\n";

		// dram_sync_obj.signal(sync_id);

		if (sync_id == 36 || sync_id == 53 || sync_id == 70 || sync_id == 87
			) { // K0_RoPE, K2_RoPE, K4_RoPE, K6_RoPE
			dram_sync_obj.signal(sync_id);
			pim_sync_obj.signal(sync_id);
		}
		else {
			if (!pim_wait_id_map[sync_id]) {
				if (cfgs.dram_enabled()) {
					dram_sync_obj.signal(sync_id);
				} else {
					pim_sync_obj.signal(sync_id);
				}
			}
			else {
				pim_sync_obj.signal(sync_id);
			}
		}
    }

	if (outstanding > 0) {
		active_cycle++;
	}
	total_cycle++;
}

void MIDAPWrapper::handle_rack(tlm_generic_payload* trans)
{
	tlm_phase phase = END_RESP;
	tlm_sync_enum reply = master->nb_transport_fw(*trans, phase, t);
	assert(reply == TLM_COMPLETED);
	uint32_t address, size;
	uint8_t* packet_data;

	if (rack_num == 0) {
		address = trans->get_address();
		size = trans->get_bst_size();
		packet_data = new uint8_t[size*sizeof(uint8_t)]; 
	}
	rack_num++;

	/* Wait until the burst data condition is satisfied */
	if (rack_num == resp_queue.front())	{
		resp_queue.pop_front();

		Packet packet{};
		packet.size = size;
		packet.address = address;
		packet.cycle = current_cycle();
 
		/* Send response to MIDAP */
		response_request(&packet);
		rack_num = 0;

		req_to_read = false;
	}

	trans->release();
}

void MIDAPWrapper::handle_wack(tlm_generic_payload* trans)
{
	tlm_phase phase = END_RESP;
	tlm_sync_enum reply = master->nb_transport_fw(*trans, phase, t);
	assert(reply == TLM_COMPLETED);

	/* Free the data memory */
	unsigned char* data_ptr = trans->get_data_ptr();
	delete[] data_ptr;

	trans->release();
}

tlm_sync_enum MIDAPWrapper::nb_transport_fw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
	peq_fw.notify(trans, phase, SC_ZERO_TIME);
	return TLM_UPDATED;
}

tlm_sync_enum MIDAPWrapper::nb_transport_bw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
    peq_bw.notify(trans, phase, SC_ZERO_TIME);
	return TLM_UPDATED;
}

void MIDAPWrapper::peq_fw_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
	run_simulator();
}

void MIDAPWrapper::peq_bw_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
	if (trans.get_command() == TLM_READ_COMMAND) {
		rack_incoming = &trans;
	} else {
		wack_incoming = &trans;
	}
}

void MIDAPWrapper::forward_trans(tlm_generic_payload *trans)
{
	tlm_phase phase = BEGIN_REQ;
	tlm_sync_enum reply = master->nb_transport_fw(*trans, phase, t);
	assert(reply == TLM_UPDATED);
	outstanding++;
}

void MIDAPWrapper::handle_packet(Packet *packet)
{
    switch(packet->type) {
        case packet_elapsed:
			set_status(RUNNING);
            break;
        case packet_read:
            handle_rw_packet(packet);
            break;
        case packet_write:
            handle_rw_packet(packet);
            break;
        case packet_bar_signal:
            handle_signal_packet(packet);
            break;
        case packet_bar_wait:
            handle_wait_packet(packet);
            break;
        case packet_terminated:
        default:
			set_status(TERMINATED);
            break;
    }
}

void MIDAPWrapper::handle_rw_packet(Packet *packet)
{
    PacketType type = packet->type;
    uint64_t addr = packet->address;
	uint32_t size = packet->size;
	unsigned int payload_num = (size > mem_req_size) ? (size / mem_req_size) : 1;

    if (type == packet_read) {
		req_to_read = true;
		resp_queue.push_back(payload_num);
		for (size_t i = 0; i < payload_num; i++) {
			add_payload(TLM_READ_COMMAND, addr+(mem_req_size*i), mem_req_size, size, NULL);
        }
    } else {
		for (size_t i = 0; i < payload_num; i++) {
			uint8_t* data = new uint8_t[mem_req_size];
			memcpy(data, (packet->data)+(mem_req_size*i), mem_req_size);	
			add_payload(TLM_WRITE_COMMAND, addr+(mem_req_size*i), mem_req_size, size, data);
		}	
    }
}

void MIDAPWrapper::handle_signal_packet(Packet *packet)
{
	uint32_t layer_id = packet->flags;
	if (npu_layer_map.find(layer_id) != npu_layer_map.end()) {
		logger->update_npu_end(npu_layer_map[layer_id], packet->cycle, packet->address);
	}
    sync_queue.push_back(packet->size);
}

void MIDAPWrapper::handle_wait_packet(Packet *packet)
{
	uint32_t layer_id = packet->flags;
	if (!map_initialized) {
		load_npu_layer_map();
		map_initialized = true;
	}
	layer = npu_layer_map[layer_id];
    unsigned int wait_id = packet->size;
	std::cout << "[MIDAP][WAIT Begin] " << layer << ", ID:" << wait_id << ", " << current_cycle() << "\n";

	SyncObject* target_sync = nullptr;

	if (!cfgs.dram_enabled()) {
		target_sync = &pim_sync_obj;
	} else {
		auto it = pim_sync_id_map.find(wait_id);
		if (it != pim_sync_id_map.end() && it->second) {
			target_sync = &pim_sync_obj;
		} else {
			target_sync = &dram_sync_obj;
		}
	}

	if (!target_sync->check_signal(wait_id)) {
		wait(*target_sync->get_event(wait_id));
	}
	
	if (npu_layer_map.find(layer_id) != npu_layer_map.end()) {
		logger->update_npu_start(layer, current_cycle());
	}

	Packet wait_packet{};
	wait_packet.cycle = current_cycle();
	response_request(&wait_packet);

	std::cout << "[MIDAP][WAIT End] " << layer << ", ID:" << wait_id << ", " << current_cycle() << "\n";
}

void MIDAPWrapper::add_payload(tlm_command cmd, uint32_t addr, uint32_t payload_size, uint32_t burst_size, uint8_t *data)
{
	tlm_generic_payload* trans = m_mm.allocate();
	trans->acquire();
	trans->set_layer(layer);
	trans->set_src_id(NPU);
	trans->set_address(addr);
	trans->set_command(cmd);
	trans->set_data_length(payload_size);
	trans->set_bst_size(burst_size);
	trans->set_layer(layer);
	trans->set_dst_id(cfgs.dram_enabled() ? DRAM : PIM);
	
	if (trans->get_command() == TLM_READ_COMMAND) {
		r_queue.push_back(trans);
	} else {
		trans->set_data_ptr(data);
		w_queue.push_back(trans);
	}
}

void MIDAPWrapper::response_request(Packet *packet)
{
    communicator->send_packet(packet);
}

void MIDAPWrapper::run_simulator()
{
	if (communicator == nullptr) {
    	communicator =  new ShmemCommunicator();
    	bool connection = communicator->prepare_connection(name.c_str(), packet_size);
		if (!connection) {
			std::cerr << "Failed to prepare shared memory connection.\n";
			exit(-1);
		}
	}

    std::stringstream command;
    command << "python3 ../../MIDAPSim/run_from_file.py"
     		<< " -l " << cfgs.get_midap_level()
     		<< " -d " << cfgs.get_compile_dir()
     		<< " -p " << cfgs.get_compile_prefix()
     		<< " -id 1 -idx 1 --debug";

    std::cout << "[MIDAP] MIDAPSim run command: " << command.str() << "\n";
    set_status(RUNNING);

    pid_t pid = fork();
    if (pid == 0) {
        int ret = system(command.str().c_str());
        if (ret < 0) {
            std::cout << "[MIDAP][ERROR] Failed to execute MIDAP.\n";
		}
        delete communicator;
        exit(-1);
    }
    communicator->wait_connection();
}

void MIDAPWrapper::set_status(Status _status){
    status = _status;
    if (_status != WAITING){
        packet_handled.notify();
    }
}

void MIDAPWrapper::elapse_cycle(int cycle)
{
	int delta_cycle = cycle - current_cycle();
	if (delta_cycle > 0) {
		wait(delta_cycle, SC_NS);
	}
}

void MIDAPWrapper::load_npu_layer_map() {
    std::ifstream file("midap_layer.json");
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open midap_layer.json.\n";
        return;
    }

    Json::Value root;
    file >> root;

    for (const auto &key : root.getMemberNames()) {
        int id = std::stoi(key);
        npu_layer_map[id] = root[key].asString();
    }
}
