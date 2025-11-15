#ifndef __HOST_H
#define __HOST_H

#include <iomanip>

#include <utilities/mm.h>
#include <utilities/logger.h>
#include <utilities/common.h>
#include <utilities/sync_object.h>
#include <utilities/configurations.h>
#include <components/trace_generator.h>
#include <components/midap_wrapper.h>

#include "tlm_utils/multi_passthrough_initiator_socket.h"
#include <tlm_utils/peq_with_cb_and_phase.h>

using namespace tlm;
using namespace sc_core;

/* External global references */
extern Configurations cfgs;
extern SyncObject dram_sync_obj;
extern SyncObject pim_sync_obj;
extern unsigned int active_host;
extern unsigned int active_dram;

class Host: public sc_module {
public:
    SC_HAS_PROCESS(Host);
    Host(sc_module_name name, Logger* logger);

    /* TLM Interface */
    tlm_utils::peq_with_cb_and_phase<Host> peq;
    tlm_utils::multi_passthrough_initiator_socket<Host> master;
    sc_in<bool> clock;    
    
    /* TLM Functions */
    void clock_negedge();
    void peq_cb(tlm_generic_payload& trans, const tlm_phase& phase);
    tlm_sync_enum nb_transport_bw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);

    void process_trace();
    void process_addertree(const std::string& layer, const std::string& cmd, unsigned int next);
    void handle_memory_trace(const std::shared_ptr<TraceGenerator::MemoryTrace>& trace);
    void handle_compute_trace(const std::shared_ptr<TraceGenerator::ComputeTrace>& trace);
    void handle_pim_trace(const std::shared_ptr<TraceGenerator::PimTrace>& trace);
    void wait_for_responses(unsigned int trans_num, const std::string& layer, bool addertree);
    tlm_generic_payload* generate_trans(TraceType type, unsigned int src, unsigned int dst, unsigned int addr,
                                    unsigned int burst_id, unsigned int burst_size,
                                    const std::string& layer, const std::string& micro_cmd = "");

    void wait_for_sync(unsigned int sync_id, unsigned int dst);
    void wait_for_compute(const std::string& layer);
    const std::string& get_op_type(const std::string& layer);
    void print_log(const tlm_generic_payload* trans);

	inline int current_cycle() const { return static_cast<int>(sc_time_stamp().to_double()/1000); }

private:
    mm m_mm;
    sc_time t{SC_ZERO_TIME};
    unsigned int cmd_id{0};
    uint32_t mem_req_size{0};
    bool host_compute_done;
    bool weight_load{false};
    uint64_t active_cycle{0};

    sc_event host_compute_done_ev;
    sc_event read_response_ev;

    std::unordered_map<std::string, unsigned int> host_profile;
    std::unordered_map<std::string, PIMCmdProfile> pim_profile;

    std::deque<std::shared_ptr<TraceGenerator::Trace>> trace_queue;
    std::deque<tlm_generic_payload*> pending_queue;
    std::deque<tlm_generic_payload*> read_response_queue;

    std::map<std::string, bool> compute_done_map;
    std::unordered_map<std::string, unsigned int> layer_tile_map;

    WaitMap pim_wait_map;
    SyncMap pim_sync_map;
    TraceGenerator trace_generator;

    std::string previous_layer = "";

    Logger* logger;
};

#endif
