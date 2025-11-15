#ifndef __CYCLE_H
#define __CYCLE_H

#include <vector>
#include <systemc>
#include <utilities/configurations.h>

extern Configurations cfgs;

struct NPUCycle {
    unsigned int start = -1;
    unsigned int end = -1;
    unsigned int compute_cycle = -1;
};

class Logger {
public:
    void update_start(const std::string& layer, DeviceType type);
    void update_end(const std::string& layer, DeviceType type);
    void update_npu_start(const std::string& layer, int start);
    void update_npu_end(const std::string& layer, int end, int compute_cycle);
    void print_info(const std::string& filename = "");

private:
    std::unordered_map<std::string, std::pair<int, int>> host_info;
    std::unordered_map<std::string, NPUCycle> npu_info;
    std::unordered_map<std::string, std::pair<int, int>> dram_info;    
    std::unordered_map<std::string, std::pair<int, int>> pim_info;
    std::unordered_map<std::string, std::pair<int, int>> pim_mcu_info;

    void print_header(std::ostream& out, const std::string& title, const std::vector<std::string>& columns);
    void print_cycle_data(std::ostream& out, const std::string& layer, int start, int end, int extra = -1);
};

#endif
