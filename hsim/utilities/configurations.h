#ifndef CONFIGURATIONS_H
#define CONFIGURATIONS_H

#include <map>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <filesystem>
#include <json.h>
#include <utilities/common.h>

typedef enum _LOG_LEVEL {
    LOG_DEBUG = 0,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    LOG_OFF
} LOG_LEVEL;

struct MemInfo {
    uint64_t shared_offset;
    uint64_t input_offset;
    uint64_t output_offset;
    uint64_t wb_offset;
    uint64_t buf_offset;
};

struct NetworkInfo {
    std::string name;
    MemInfo meminfo;
    std::string out_path;
    std::string prefix;
    bool precompiled;
};

struct CompileInfo {
    bool quantized;
    std::string midap_compiler;
    unsigned packet_size;
    unsigned midap_level;
    uint32_t fmem_bank_num;
    uint32_t fmem_bank_size;
    uint32_t cim_num;
    uint32_t wmem_size;
    uint32_t ewmem_size;
    NetworkInfo net_info;
    std::string additional_flags;
};

struct DRAMInfo {
    uint32_t channels;
    uint32_t capacity;
    std::string backend;
    std::string type;
    std::string config;
    std::string preset;
    double freq;
};

struct PIMInfo {
    uint32_t channels;
    uint32_t capacity;
    std::string backend;
    std::string type;
    std::string config;
    std::string preset;
    double freq;
};

struct DelayCount {
    unsigned int delay;
    unsigned int count;
};

struct PIMCmdProfile {
    std::unordered_map<std::string, DelayCount> cmd_map;
    std::vector<std::string> ordered_cmd_name;
};

class Configurations {
public:
    Configurations();

    Json::Value parse_json(const std::string& file_name);

    void init_dram();
    void init_system();
    void init_compiler();
    void init_sync_wait_map();
    void init_configurations();
    void compile_network();
    void init_host_profile();
    void init_pim_profile(const std::string& pim_type);
    
    bool pim_enabled()  { return enable_pim; }
    bool dram_enabled() { return enable_dram; }
    bool mcu_enabled()  { return enable_mcu; }

    uint32_t get_scenario() { return scenario_id; }
    uint32_t get_packet_size() { return compileinfo.packet_size; }
    uint32_t get_midap_level() { return compileinfo.midap_level; }
    uint32_t get_dram_req_size() { return dram_req_size; }
    uint64_t get_dram_capacity() { return draminfo.capacity; }
    uint32_t get_dram_channels() { return draminfo.channels; }
    uint32_t get_pim_channels() { return piminfo.channels; }

    MemInfo get_meminfo() { return compileinfo.net_info.meminfo; }
    NetworkInfo get_netinfo() { return compileinfo.net_info; }
    SyncMap get_host_sync_map() { return host_sync_map; }
    WaitMap get_host_wait_map() { return host_wait_map; }
    SyncMap get_pim_sync_map() { return pim_sync_map; }
    WaitMap get_pim_wait_map() { return pim_wait_map; }

    double get_dram_freq() { return draminfo.freq; }
    double get_pim_freq() { return piminfo.freq; }

    double get_host_freq() { return host_frequency; }
    double get_mcu_freq() { return mcu_frequency; }

    std::string get_cycle_log_file() { return cycle_log_file; }

    std::string get_pim_config() { return piminfo.config; }
    std::string get_pim_type() { return piminfo.type; }
    std::string get_pim_preset() { return piminfo.preset; }

    std::string get_dram_config() { return draminfo.config; }
    std::string get_dram_type() { return draminfo.type; }
    std::string get_dram_preset() { return draminfo.preset; }
    std::string get_dram_backend() { return draminfo.backend; }

    std::string get_memory_structure() { return memory_structure; }

    std::string get_net_name() { return compileinfo.net_info.name; }
    std::string get_compile_dir() { return compileinfo.net_info.out_path; }
    std::string get_compile_prefix() { return compileinfo.net_info.prefix; }
    std::string get_target_model() { return target_model; }
    
    const std::unordered_map<unsigned int, bool>& get_pim_sync_id_map() { return pim_sync_id_map; }
    const std::unordered_map<unsigned int, bool>& get_pim_wait_id_map() { return pim_wait_id_map; }
    const std::unordered_map<std::string, unsigned int>& get_layer_tile_map() { return layer_tile_map; }
    const std::unordered_map<std::string, unsigned int>& get_host_profile() { return host_profile; }
    const std::unordered_map<std::string, PIMCmdProfile>& get_pim_profile() { return pim_profile; }
    const std::tuple<unsigned int, unsigned int>& get_mode_change_delay() { return mode_change_delay; }
    void load_sync_wait_info(const std::string& file_name, BaseMap& info_map);
    void print_configurations();

    unsigned int get_attn_head_num() { return num_attn_heads; }

private:
    bool enable_pim;
    bool enable_dram;    
    bool enable_mcu;

    uint32_t scenario_id;
    uint32_t dram_req_size;
    std::string memory_structure;
    std::string cycle_log_file;
    CompileInfo compileinfo;
    DRAMInfo draminfo;
    PIMInfo piminfo;

    unsigned int dram_to_pim_delay;
    unsigned int pim_to_dram_delay;    
    unsigned int input_sequence_length;
    unsigned int output_sequence_length;
    unsigned int decoder_block;
    unsigned int num_attn_heads;
    double host_frequency;
    double mcu_frequency;

    std::string target_model;

    std::map<std::string, int> dram_list;
    std::unordered_map<std::string, unsigned int> layer_tile_map;

    std::unordered_map<std::string, unsigned int> host_profile;
    std::unordered_map<std::string, PIMCmdProfile> pim_profile;
    std::unordered_map<unsigned int, bool> pim_sync_id_map;
    std::unordered_map<unsigned int, bool> pim_wait_id_map;
    std::tuple<unsigned int, unsigned int> mode_change_delay;    

    SyncMap host_sync_map;
    WaitMap host_wait_map;
    SyncMap pim_sync_map;
    WaitMap pim_wait_map;
};

#endif //CONFIGURATIONS_H
