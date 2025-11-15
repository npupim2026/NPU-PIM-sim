#ifndef MCU_H
#define MCU_H

#include <utilities/mm.h>
#include <utilities/common.h>
#include <utilities/logger.h>
#include <utilities/configurations.h>
#include <utilities/sync_object.h>
#include <components/trace_generator.h>

#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"
#include <tlm_utils/peq_with_cb_and_phase.h>

using namespace tlm;
using namespace sc_core;

extern Configurations cfgs;
extern SyncObject pim_sync_obj;
extern bool is_pim_busy;
extern sc_event pim_compute_done_ev;
extern unsigned int active_host;
extern unsigned int active_dram;
extern unsigned int active_core;

class Mcu: public sc_module
{
public:
    SC_HAS_PROCESS(Mcu);
    Mcu(sc_module_name name, Logger* logger);
    ~Mcu();
    
    /* TLM interfaces */
	sc_in<bool> clock;
    tlm_utils::peq_with_cb_and_phase<Mcu> peq;
	tlm_utils::simple_initiator_socket<Mcu> master;
	tlm_utils::simple_target_socket<Mcu> slave;

    /* Main simulation */
    void process_trace();
    void clock_negedge();

    /* Transport interface */
	tlm_sync_enum nb_transport_fw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);
    tlm_sync_enum nb_transport_bw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);
    void peq_cb(tlm_generic_payload& trans, const tlm_phase& phase);
    
    /* Processing */
    void wait_for_responses(unsigned int trans_num, const std::string& layer, bool addertree);
    void wait_for_dependencies(const std::string& layer);

    void process_addertree(const std::string& layer, const std::string& cmd);
    void handle_pim_trans(tlm_generic_payload& trans);
    tlm_generic_payload* generate_trans(TraceType type, unsigned int src, const std::string& op,
                                        const std::string& layer, unsigned int burst_id, unsigned int burst_size);
    void generate_pim_trans(const std::string& layer, const PIMCmdProfile& profile, const std::vector<std::string>& cmd_list,
                            unsigned int tile_num);

    void wait_for_sync(unsigned int id);
    void process_read_responses(unsigned int size);
    void print_log(const tlm_generic_payload* trans);

private:
    mm m_mm;
    sc_event read_response_ev;
    sc_event trans_arrival_ev;
    sc_time t{SC_ZERO_TIME};
    SyncMap pim_sync_map;
    WaitMap pim_wait_map;

    std::deque<tlm_generic_payload*> fw_queue;
    std::deque<tlm_generic_payload *> trans_queue;
    std::deque<tlm_generic_payload*> read_response_queue;

    std::unordered_map<std::string, unsigned int> host_profile;
    std::unordered_map<std::string, PIMCmdProfile> pim_profile;
    std::unordered_map<std::string, unsigned int> layer_tile_map;

    uint32_t mem_req_size{0};
    unsigned int cmd_id{0};
    std::string previous_layer = "";

    Logger* logger;
};
#endif