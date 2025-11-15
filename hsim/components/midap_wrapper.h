#ifndef __MIDAP_CORE_H
#define __MIDAP_CORE_H

#include <utilities/mm.h>
#include <utilities/logger.h>
#include <utilities/hsim_packet.h>
#include <utilities/configurations.h>
#include <utilities/sync_object.h>
#include <utilities/shmem_communicator.h>

#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"
#include "tlm_utils/peq_with_cb_and_phase.h"

extern Configurations cfgs;
extern SyncObject dram_sync_obj;
extern SyncObject pim_sync_obj;
extern unsigned int active_core;
extern unsigned int active_dram;

using namespace tlm;
using namespace sc_core;

enum Status { INIT, RUNNING, WAITING, TERMINATED };

class MIDAPWrapper: public sc_module {
public:
    /* Socket & Port */
	tlm_utils::simple_initiator_socket<MIDAPWrapper> master;
    tlm_utils::simple_target_socket<MIDAPWrapper> slave;
    sc_in<bool> clock;
    
    tlm_utils::peq_with_cb_and_phase<MIDAPWrapper> peq_fw;
    tlm_utils::peq_with_cb_and_phase<MIDAPWrapper> peq_bw;

    MIDAPWrapper(sc_module_name name, Logger* logger);
    ~MIDAPWrapper();

    SC_HAS_PROCESS(MIDAPWrapper);

    tlm_sync_enum nb_transport_fw(tlm_generic_payload& trans,
                                  tlm_phase& phase,
                                  sc_time& t);

	tlm_sync_enum nb_transport_bw(tlm_generic_payload& trans,
                                  tlm_phase& phase,
                                  sc_time& t);

    void periodic_process();
    void clock_posedge();
    void clock_negedge();
    inline int current_cycle() const { return static_cast<int>(sc_time_stamp().to_double()/1000); }

private:
    void run_simulator();
    void handle_packet(Packet *packet);
    void handle_rack(tlm_generic_payload* trans);
    void handle_wack(tlm_generic_payload* trans);
    void handle_rw_packet(Packet *packet);
    void handle_signal_packet(Packet *packet);
    void handle_wait_packet(Packet *packet);
    void forward_trans(tlm_generic_payload *trans);
    void add_payload(tlm_command cmd, uint32_t addr, uint32_t payload_size, uint32_t burst_size, uint8_t *data);
	void peq_fw_cb(tlm_generic_payload& trans, const tlm_phase& phase);
    void peq_bw_cb(tlm_generic_payload& trans, const tlm_phase& phase);
    void response_request(Packet *packet);
    void elapse_cycle(int cycle);
    void load_npu_layer_map();

    mm m_mm;
    sc_time t{SC_ZERO_TIME};
    std::string name;
    Status status;
    ShmemCommunicator *communicator = nullptr;

	uint32_t rack_num{0};
    uint32_t wack_num{0};
    uint32_t packet_size;
    uint32_t outstanding{0};
    uint32_t mem_req_size;
    std::string layer;

    bool req_to_read{false};

    uint64_t active_cycle{0};
    uint64_t total_cycle{0};

    std::deque<tlm_generic_payload*> w_queue;
    std::deque<tlm_generic_payload*> r_queue;
    std::deque<tlm_generic_payload*> wack_queue;
    std::deque<tlm_generic_payload*> rack_queue;
    std::deque<tlm_generic_payload*> pending_queue;
    std::deque<unsigned int> resp_queue;
    std::deque<unsigned int> w_resp_queue;
    std::deque<unsigned int> sync_queue;

    tlm_generic_payload* wack_incoming{NULL};
    tlm_generic_payload* rack_incoming{NULL};

    sc_event packet_handled;

    void set_status(Status _status);
    bool write_pending{false};
    bool map_initialized{false};

    SyncMap pim_sync_map;

    std::map<int, std::string> npu_layer_map;
    std::unordered_map<unsigned int, bool> pim_sync_id_map;
    std::unordered_map<unsigned int, bool> pim_wait_id_map;
    Logger* logger;
};

#endif
