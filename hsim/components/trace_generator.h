#ifndef __TRACE_GENERATOR_H
#define __TRACE_GENERATOR_H

#include <systemc.h>
#include <unordered_set>
#include <utilities/common.h>
#include <utilities/configurations.h>

extern Configurations cfgs;

enum class TraceType {
    WRITE,
    READ,
    COMPUTE,
    PIM,
    TERMINATE
};

class TraceGenerator {
public:   
    struct Trace {
        virtual ~Trace() = default;
        TraceType type;
        std::string layer;
        Trace(TraceType t, std::string l) : type(t), layer(l) {}
    };

    struct MemoryTrace : public Trace {
        unsigned int id, src, dst, size;
        uint64_t address;
        MemoryTrace(TraceType t, unsigned int i, unsigned int s, unsigned int d, unsigned int z, uint64_t a, std::string l)
        : Trace(t, l), id(i), src(s), dst(d), size(z), address(a) {}  
    };

    struct ComputeTrace : public Trace {
        unsigned int device;
        ComputeTrace(TraceType t, unsigned int d, std::string l)
        : Trace(t, l), device(d) {}  
    };

    struct SimTrace : public Trace {
        SimTrace(TraceType t, std::string l)
        : Trace(t, l) {}  
    };

    struct PimTrace : public Trace {
        std::string cmd;
        unsigned int next;
        PimTrace(TraceType t, std::string l, std::string c, unsigned int n)
        : Trace(t, l), cmd(c), next(n) {}  
    };

    TraceGenerator();
    std::deque<std::shared_ptr<Trace>> trace_queue;
    std::unordered_set<std::string> layers;

    std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_trace();
    
    void add_trace(TraceType type, const std::string& layer, unsigned int device);
    void add_gemv_trace(const std::string& layer, const std::vector<std::string>& cmds, unsigned int next);
    void add_rope_trace(const std::string& layer, const std::vector<std::string>& cmds);
    void add_memory_trace(TraceType type, const std::string& layer, DeviceType src, const std::vector<DeviceType>& dst_list);
    void add_host_trace(const std::string& layer, const std::vector<DeviceType>& r_targets, const std::vector<DeviceType>& w_targets);

private:
    SyncMap host_sync_map;
    WaitMap host_wait_map;
};

#endif