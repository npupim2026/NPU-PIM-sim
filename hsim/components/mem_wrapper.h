#ifndef MEM_WRAPPER_H
#define MEM_WRAPPER_H

#include <utilities/common.h>
#include <utilities/logger.h>
#include <utilities/sync_object.h>
#include <utilities/configurations.h>
#include <components/trace_generator.h>

#include "tlm_utils/simple_target_socket.h"
#include <tlm_utils/peq_with_cb_and_phase.h>

#include "Bridge.h"
#include "bridge.h"

extern Configurations cfgs;
extern SyncObject dram_sync_obj;
extern SyncObject pim_sync_obj;
extern unsigned int active_host;
extern unsigned int active_core;
extern unsigned int active_dram;

using namespace tlm;
using namespace sc_core;

class MEMWrapper: public sc_module
{
public:
    SC_HAS_PROCESS(MEMWrapper);

	sc_in<bool> clock;
	tlm_utils::peq_with_cb_and_phase<MEMWrapper> peq;

    std::deque<tlm_generic_payload *> fw_queue;	
	std::deque<tlm_generic_payload *> bw_queue;
	std::deque<tlm_generic_payload *> mem_request;

	tlm_utils::simple_target_socket<MEMWrapper> slave;

    MEMWrapper(sc_module_name name, Logger* logger);
    ~MEMWrapper();

	void simulate_dram();
	void clock_posedge();
    void clock_negedge();
	tlm_sync_enum nb_transport_fw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);
	void peq_cb(tlm_generic_payload& trans, const tlm_phase& phase);
	inline int current_cycle() const { return static_cast<int>(sc_time_stamp().to_double()/1000); }

	void complete_write_request(tlm::tlm_generic_payload* trans);
	void signal_sync(BaseMap& map, const std::string& layer);

private:
	void init_dram();
    void backward_trans(tlm_generic_payload* trans, bool read);
	tlm_generic_payload* gen_trans(uint64_t addr, tlm_command cmd, uint32_t size);
	unsigned int find_sync_id(std::string layer);

	uint64_t active_cycle;
	uint64_t total_cycle;
	uint32_t m_outstanding{0};
	std::string config_name;
	sc_time t{SC_ZERO_TIME};
	uint64_t sync_id;
	uint32_t dram_req_size;

	ramulator2::Bridge *ramulator2_bridge;
	dramsim3::Bridge *dramsim3_bridge;

	SyncMap host_sync_map;
	SyncMap pim_sync_map;
	Logger* logger;
};
#endif //MEM_CTRL_H
