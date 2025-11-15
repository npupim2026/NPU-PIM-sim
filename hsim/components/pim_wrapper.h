#ifndef PIM_WRAPPER_H
#define PIM_WRAPPER_H

#include <utilities/logger.h>
#include <utilities/configurations.h>
#include <components/host.h>
// #include "tlm_utils/simple_target_socket.h"
#include "tlm_utils/multi_passthrough_target_socket.h"
#include <tlm_utils/peq_with_cb_and_phase.h>

#include "bridge.h"

extern Configurations cfgs;
extern bool is_pim_busy;
extern sc_event pim_compute_done_ev;

using namespace tlm;
using namespace sc_core;

enum class PimMode { COMPUTE, MEMORY };

class PIMWrapper: public sc_module
{
public:
    SC_HAS_PROCESS(PIMWrapper);

	sc_in<bool> clock;
	tlm_utils::peq_with_cb_and_phase<PIMWrapper> peq;
	// tlm_utils::simple_target_socket<PIMWrapper> slave;
	tlm_utils::multi_passthrough_target_socket<PIMWrapper> slave;

    PIMWrapper(sc_module_name name, Logger* logger);
    ~PIMWrapper();

	void clock_posedge();
    void clock_negedge();
	// tlm_sync_enum nb_transport_fw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);
	tlm_sync_enum nb_transport_fw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);

	void peq_cb(tlm_generic_payload& trans, const tlm_phase& phase);

    void signal_sync(BaseMap& target_map, const std::string& layer);
	void send_response(tlm::tlm_generic_payload* trans, unsigned int dst_id);
	void complete_write_request(tlm::tlm_generic_payload* trans);
	
	void simulate_partitioned_pim();
	void simulate_unified_pim();
	void init_pim_dram();

	void process_mode_transition(PimMode mode);
	void send_transaction(tlm_generic_payload* trans, bool is_compute);

	inline int current_cycle() const { return static_cast<int>(sc_time_stamp().to_double()/1000); }

private:
	sc_time t{SC_ZERO_TIME};
	Logger* logger;

	SyncMap pim_sync_map;
    SyncMap host_sync_map;

	std::tuple<unsigned int, unsigned int> mode_change_delay;
	dramsim3::Bridge *bridge;

	std::deque<tlm_generic_payload *> pending_queue;	
	std::deque<tlm_generic_payload *> memory_completed_queue;
	std::deque<tlm_generic_payload *> compute_completed_queue;

	PimMode pim_mode;

	unsigned int mode_change_count{0};
	unsigned int pending_count{0};

	bool memory_busy{false};
	bool compute_busy{false};

	unsigned int memory_busy_count{0};
    unsigned int compute_busy_count{0};
};
#endif
