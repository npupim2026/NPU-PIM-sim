#include "trace_generator.h"

#include <scenarios/baseline.h>
#include <scenarios/partitioned.h>
#include <scenarios/unified.h>

TraceGenerator::TraceGenerator() {
    host_sync_map = cfgs.get_host_sync_map();
    host_wait_map = cfgs.get_host_wait_map();
};

std::deque<std::shared_ptr<TraceGenerator::Trace>>& TraceGenerator::generate_trace()
{
    unsigned int scenario = cfgs.get_scenario();
    if (cfgs.get_memory_structure() == "baseline") {
        return generate_baseline(*this);
    } else if (cfgs.get_memory_structure() == "partitioned") {
        return generate_partitioned(*this);
    } else {
        return generate_unified(*this);
    }
    throw std::runtime_error("Unhandled trace configuration.");
}

void TraceGenerator::add_memory_trace(TraceType type, const std::string& layer, DeviceType src, const std::vector<DeviceType>& dst_list)
{
    for (const auto& dst : dst_list) {
        BaseMap* base_map = nullptr;

        if (src == HOST) {
            base_map = (type == TraceType::READ) ? &host_wait_map : &host_sync_map;
        }

        bool found = false;

        for (auto it = base_map->begin(); it != base_map->end(); ++it) {
            if (layer == "RoPE_") {
                if (it->first.find(layer) != std::string::npos) {
                    found = true;
                    const auto& [id, size, address] = it->second;
                    trace_queue.push_back(std::make_shared<MemoryTrace>(type, id, src, dst, size, address, it->first));
                }
            }
            else if (it->first == layer) {
                found = true;
                const auto& [id, size, address] = it->second;
                trace_queue.push_back(std::make_shared<MemoryTrace>(type, id, src, dst, size, address, it->first));
            }
        }

        if (!found) {
            std::cerr << "[TraceGenerator][ERROR] No entries found for " << layer << " in map!" << std::endl;
        }
    }
}

void TraceGenerator::add_trace(TraceType type, const std::string& layer, unsigned int device) {
    if (type == TraceType::COMPUTE) {
        trace_queue.push_back(std::make_shared<ComputeTrace>(type, device, layer));
    } else if (type == TraceType::TERMINATE) {
        trace_queue.push_back(std::make_shared<SimTrace>(type, layer));
    }
}

void TraceGenerator::add_host_trace(const std::string& layer, const std::vector<DeviceType>& r_targets, const std::vector<DeviceType>& w_targets)
{
    if (!r_targets.empty()) { add_memory_trace(TraceType::READ, layer, HOST, r_targets); }
    add_trace(TraceType::COMPUTE, layer, HOST);
    if (!w_targets.empty()) { add_memory_trace(TraceType::WRITE, layer, HOST, w_targets); }
}

void TraceGenerator::add_gemv_trace(const std::string& layer, const std::vector<std::string>& cmds, unsigned int next)
{
    std::unordered_map<std::string, unsigned int> tile_map = cfgs.get_layer_tile_map();
    unsigned int tile_num = tile_map[layer];

    for (int i = 0; i < tile_num; i++) {
        for (const auto& cmd : cmds) {
            trace_queue.push_back(std::make_shared<PimTrace>(TraceType::PIM, layer, cmd, next));
        }
    }
    trace_queue.push_back(std::make_shared<PimTrace>(TraceType::PIM, layer, "addertree", next));
}

void TraceGenerator::add_rope_trace(const std::string& layer, const std::vector<std::string>& cmds)
{
    std::unordered_map<std::string, unsigned int> tile_map = cfgs.get_layer_tile_map();
    unsigned int tile_num = tile_map[layer];

    for (int i = 0; i < tile_num; i++) {
        for (const auto& cmd : cmds) {
            trace_queue.push_back(std::make_shared<PimTrace>(TraceType::PIM, layer, cmd, PIM));
        }
    }
}