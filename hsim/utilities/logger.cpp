#include "logger.h"

void Logger::update_start(const std::string& layer, DeviceType type)
{
    int cycle = static_cast<int>(sc_core::sc_time_stamp().to_double() / 1000);
    switch (type) {
        case DeviceType::HOST:
            if (host_info.find(layer) == host_info.end())
                host_info[layer].first = cycle;
            break;
        case DeviceType::NPU:
            if (npu_info.find(layer) == npu_info.end())
                npu_info[layer].start = cycle;
            break;
        case DeviceType::PIM:
            if (pim_info.find(layer) == pim_info.end())
                pim_info[layer].first = cycle;
            break;
        case DeviceType::DRAM:
            if (dram_info.find(layer) == dram_info.end())
                dram_info[layer].first = cycle;
            break;            
        case DeviceType::MCU:
            if (pim_mcu_info.find(layer) == pim_mcu_info.end())
                pim_mcu_info[layer].first = cycle;
            break;
    }
}

void Logger::update_end(const std::string& layer, DeviceType type)
{
    int cycle = static_cast<int>(sc_core::sc_time_stamp().to_double() / 1000);
    switch (type) {
        case DeviceType::HOST:
            host_info[layer].second = cycle;
            break;
        case DeviceType::PIM:
            pim_info[layer].second = cycle;
            break;
        case DeviceType::DRAM:
            dram_info[layer].second = cycle;
            break;            
        case DeviceType::MCU:
            pim_mcu_info[layer].second = cycle;
            break;
    }
}

void Logger::update_npu_start(const std::string& layer, int start)
{
    int cycle = static_cast<int>(sc_core::sc_time_stamp().to_double() / 1000);
    auto& cycle_info = npu_info[layer];

	if (layer == "D_MHA_out") {
        cycle_info.start = start;
	}
	
	else {
	    if (cycle_info.start == -1) {
        	if (layer == "D_Q0")
    	        cycle_info.start = cycle;
	        else {
            	cycle_info.start = start;
        	}
    	}
	}
}

void Logger::update_npu_end(const std::string& layer, int end, int compute_cycle)
{
    npu_info[layer].end = end;
    npu_info[layer].compute_cycle = compute_cycle;
}

void Logger::print_info(const std::string& filename)
{
    std::ostream* out_stream = &std::cout;
    std::ofstream ofs;

    if (!filename.empty()) {
        ofs.open(filename);
        if (!ofs.is_open()) {
            std::cerr << "Failed to open file: " << filename << "\n";
            return;
        }
        out_stream = &ofs;
    }

    std::ostream& out = *out_stream;

    print_header(out, "Host Execution Cycle Info", {"Layer", "Start", "End", "Total"});
    for (const auto& [layer, cycle] : host_info) {
        if (cycle.first == -1) continue;
        print_cycle_data(out, layer, cycle.first, cycle.second);
    }

    print_header(out, "NPU Execution Cycle Info", {"Layer", "Start", "End", "DRAM delay", "Total"});
    for (const auto& [layer, cycle] : npu_info) {
        print_cycle_data(out, layer, cycle.start, cycle.end, cycle.compute_cycle);
    }

    if (cfgs.pim_enabled()) {
        print_header(out, "PIM Execution Cycle Info", {"Layer", "Start", "End", "Total"});
        for (const auto& [layer, cycle] : pim_info) {
            print_cycle_data(out, layer, cycle.first, cycle.second);
        }
    }

    if (cfgs.mcu_enabled()) {
        print_header(out, "MCU Execution Cycle Info", {"Layer", "Start", "End", "Total"});
        for (const auto& [layer, cycle] : pim_mcu_info) {
            print_cycle_data(out, layer, cycle.first, cycle.second);
        }
    }

    if (ofs.is_open()) {
        std::cout << "Cycle info saved to file: " << filename << "\n";
    }
}

void Logger::print_header(std::ostream& out, const std::string& title, const std::vector<std::string>& columns)
{
    out << "\n===== " << title << " =====\n";
    for (const auto& col : columns)
        out << col << " ";
    out << "\n" << std::string(60, '-') << "\n";
}


void Logger::print_cycle_data(std::ostream& out, const std::string& layer, int start, int end, int extra)
{
    int duration = (end > 0) ? (end - start) : -1;
        out << layer << " "
            << start << " "
            << (end > 0 ? std::to_string(end) : "N/A") << " ";
        if (extra >= 0) {
            out << duration-extra << " ";
        }
        out << (duration >= 0 ? std::to_string(duration) : "N/A") << "\n";    
}
