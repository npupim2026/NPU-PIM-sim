#include "configurations.h"

Configurations::Configurations()
{};

void Configurations::init_configurations()
{
    init_system();
    init_dram();
    init_compiler();
    compile_network();
    init_sync_wait_map();
}

void Configurations::print_configurations()
{
    std::cout << "\n========== Simulation Configuration ==========\n";

    std::cout << "Frequency (GHz):\n";
    std::cout << "  ∙ Host (BP) : " << host_frequency << "\n";
    if (enable_mcu) { std::cout << "  ∙ MCU       : " << mcu_frequency << "\n"; }

    std::cout << "Components:\n";
    std::cout << "  ∙ Enable DRAM : " << (enable_dram ? "Yes" : "No") << "\n";
    std::cout << "  ∙ Enable PIM  : " << (enable_pim  ? "Yes" : "No") << "\n";
    std::cout << "  ∙ Enable MCU  : " << (enable_mcu  ? "Yes" : "No") << "\n";

    std::cout << "\nMapping Scenario:\n";
    std::cout << "  ∙ Scenario ID : " << scenario_id << "\n";

    std::cout << "\nDRAM-PIM Structure: " << memory_structure << "\n";

    std::cout << "\nDRAM Configuration:\n";
    if (enable_dram) {
        std::cout << "  ∙ Type              : " << draminfo.type << "_" << draminfo.config << "\n";
        std::cout << "  ∙ Backend           : " << draminfo.backend << "\n";
        std::cout << "  ∙ Timing Preset     : " << draminfo.preset << "\n";
        std::cout << "  ∙ Channels          : " << draminfo.channels << "\n";
        std::cout << "  ∙ Capacity (GB)     : " << draminfo.capacity << "\n";
        std::cout << "  ∙ Frequency (GHz)   : " << draminfo.freq << "\n";
    } else {
        std::cout << "  ∙ [Disabled]\n";
    }

    std::cout << "\nPIM Configuration:\n";
    if (enable_pim) {
        std::cout << "  ∙ Type              : " << piminfo.type << "_" << piminfo.config << "\n";
        std::cout << "  ∙ Backend           : " << piminfo.backend << "\n";
        std::cout << "  ∙ Timing Preset     : " << piminfo.preset << "\n";
        std::cout << "  ∙ Channels          : " << piminfo.channels << "\n";
        std::cout << "  ∙ Capacity (GB)     : " << piminfo.capacity << "\n";
        std::cout << "  ∙ Frequency (GHz)   : " << piminfo.freq << "\n";
    } else {
        std::cout << "  [Disabled]\n";
    }
    std::cout << "\n Workload:\n";
    std::cout << "  ∙ Input Sequence Length  : " << input_sequence_length << "\n";
    std::cout << "  ∙ Output Sequence Length : " << output_sequence_length << "\n";
    std::cout << "  ∙ Decoder Block : " << decoder_block << "\n";
    std::cout << "==============================================\n\n";
}

void Configurations::init_system()
{
    Json::Value root = parse_json("system");
    const Json::Value arch = root["architecture"];
    const Json::Value work = root["workload"];
    const Json::Value freq = root["frequency"];

    memory_structure = arch["memory_structure"].asString();
    enable_mcu = arch["enable_mcu"].asBool();

    target_model = work["model"].asString();
    num_attn_heads  = work["attention_head"].asUInt();
    scenario_id = work["scenario_id"].asUInt();
    input_sequence_length = work["input_sequence_length"].asUInt();
    output_sequence_length = work["output_sequence_length"].asUInt();
    decoder_block = work["decoder_block"].asUInt();

    host_frequency = freq["host"].asDouble();
    if (enable_mcu) { mcu_frequency = freq["mcu"].asDouble(); }

    cycle_log_file = root["cycle_log_file"].asString();

    if (memory_structure == "baseline") {
        enable_dram = true;
        enable_pim = false;
    } else if (memory_structure == "partitioned") {
        enable_dram = true;
        enable_pim = true;
    } else if (memory_structure == "unified") {
        enable_dram = false;
        enable_pim = true;
    } else {
        throw std::runtime_error("Invalid memory_structure: " + memory_structure);
    }
}

void Configurations::init_dram()
{
    Json::Value root = parse_json("memory");

    dram_req_size = root["request_size"].asUInt(); /* Bytes */

    const Json::Value dram = root["dram"];
    draminfo.backend = dram["backend"].asString();
    draminfo.type = dram["memory_type"].asString();
    draminfo.config = dram["chip_config"].asString();
    draminfo.preset = dram["timing_preset"].asString();
    draminfo.channels = dram["num_channels"].asUInt();
    draminfo.capacity = dram["capacity"].asUInt();
    draminfo.freq = dram["clock_freq"].asDouble();

    const Json::Value pim = root["pim"];
    piminfo.backend = pim["backend"].asString();
    piminfo.type = pim["memory_type"].asString();
    piminfo.config = pim["chip_config"].asString();
    piminfo.preset = pim["timing_preset"].asString();    
    piminfo.channels = pim["num_channels"].asUInt();
    piminfo.capacity = pim["capacity"].asUInt();
    piminfo.freq = pim["clock_freq"].asDouble();

    init_host_profile();
    init_pim_profile(piminfo.type);
}

void Configurations::init_host_profile()
{
    Json::Value root = parse_json("profile/host");
    const Json::Value& host = root["host"];
    for (const auto& op : host.getMemberNames()) {
        host_profile[op] = host[op].asUInt();
    }
}

void Configurations::init_pim_profile(const std::string& pim_type)
{
    init_host_profile();
    
    /* PIM */
    std::string file_name = "profile/" + pim_type;
    
    if (input_sequence_length == 1024) {
        file_name += "_1024";   
    } else if (input_sequence_length == 512) {
        file_name += "_512";
    }

    if (target_model == "gemma2-9B") {
        file_name += "_9b";
    }

    Json::Value root = parse_json(file_name);
    
    const Json::Value mode_delay = root["mode_change_delay"];
    pim_to_dram_delay = mode_delay["PIM_to_DRAM"].asUInt();
    dram_to_pim_delay = mode_delay["DRAM_to_PIM"].asUInt();

    mode_change_delay = std::make_tuple(pim_to_dram_delay, dram_to_pim_delay);
    
    const Json::Value& operations = root["operations"];
    const Json::Value& layers = root["layers"];

    for (const auto& name : layers.getMemberNames()) {
        std::string operation = layers[name].asString();
        const Json::Value& cmd_array = operations[operation];
        PIMCmdProfile profile;

        for (const auto& entry : cmd_array) {
            if (entry.isMember("output_tile")) {
                if (!entry["output_tile"].isUInt()) {
                    throw std::runtime_error("Invalid 'output_tile' value in operation '" + operation + "'");
                }
                layer_tile_map[name] = entry["output_tile"].asUInt();
                continue;
            }
            if (!entry.isMember("name") || !entry["name"].isString()) {
                throw std::runtime_error("Missing or invalid 'name' in operation '" + operation + "'");
            }
            if (!entry.isMember("latency") || !entry["latency"].isUInt()) {
                throw std::runtime_error("Missing or invalid 'latency' in operation '" + operation + "'");
            }
            if (!entry.isMember("count") || !entry["count"].isUInt()) {
                throw std::runtime_error("Missing or invalid 'count' in operation '" + operation + "'");
            }
            std::string cmd = entry["name"].asString();
            profile.cmd_map[cmd] = { entry["latency"].asUInt(), entry["count"].asUInt() };
            profile.ordered_cmd_name.push_back(cmd);
        }
        pim_profile[name] = profile;
    }
}

void Configurations::init_compiler()
{
    Json::Value compiler = parse_json("compiler");

    compileinfo.quantized = compiler["quantized"].asBool();
    compileinfo.additional_flags = compiler["additional_flags"].asString();

    compileinfo.midap_compiler = compiler["layer_compiler"].asString();
    if ((compileinfo.midap_compiler != "MIN_DRAM_ACCESS")
        && (compileinfo.midap_compiler != "HIDE_DRAM_LATENCY") && (compileinfo.midap_compiler != "DOUBLE_BUFFER")) {
        std::cout << "Invalid compiler option - " << compileinfo.midap_compiler << std::endl;
        exit(-1);
    }

    compileinfo.packet_size = compiler["packet_size"].asUInt();
    compileinfo.midap_level = compiler["midap_level"].asUInt();
    compileinfo.fmem_bank_num = compiler["fmem_bank_num"].asUInt();
    compileinfo.fmem_bank_size = compiler["fmem_bank_size"].asUInt();
    compileinfo.cim_num = compiler["cim_num"].asUInt();
    compileinfo.wmem_size = compiler["wmem_size"].asUInt();
    compileinfo.ewmem_size = compiler["ewmem_size"].asUInt();

    const Json::Value network = compiler["network"];
    auto it = network.begin();

    if (it->isObject()) {
        compileinfo.net_info.name = (*it)["net_name"].asString();
        std::string shared_offset = (*it)["shared_offset"].asString();
        std::string input_offset = (*it)["input_offset"].asString();
        std::string output_offset = (*it)["output_offset"].asString();
        std::string wb_offset = (*it)["wb_offset"].asString();
        std::string buf_offset = (*it)["buf_offset"].asString();
        std::stringstream shared_off_stream(shared_offset);
        std::stringstream input_off_stream(input_offset);
        std::stringstream output_off_stream(output_offset);
        std::stringstream wb_off_stream(wb_offset);
        std::stringstream buf_off_stream(buf_offset);

        shared_off_stream >> std::hex >> compileinfo.net_info.meminfo.shared_offset;
        input_off_stream >> std::hex >> compileinfo.net_info.meminfo.input_offset;
        output_off_stream >> std::hex >> compileinfo.net_info.meminfo.output_offset;
        wb_off_stream >> std::hex >> compileinfo.net_info.meminfo.wb_offset;
        buf_off_stream >> std::hex >> compileinfo.net_info.meminfo.buf_offset;

        std::stringstream temp;
        compileinfo.net_info.out_path = (*it)["path"].asString();
        temp << ROOT_PATH << compileinfo.net_info.out_path << "/";
        compileinfo.net_info.out_path = temp.str();

        compileinfo.net_info.prefix = (*it)["prefix"].asString();
        compileinfo.net_info.precompiled = (*it)["precompiled"].asBool();
    } 
}

void Configurations::compile_network()
{
    if (compileinfo.net_info.precompiled == false) {
        std::stringstream command;
        command << "cd " << std::filesystem::path(ROOT_PATH) / "MIDAPSim" << " ; "
                << "python tools/test_system.py"
                << " -n " << compileinfo.net_info.name
                << " -d DMA -fs -so "
                << (compileinfo.quantized ? "-q " : "")
                << compileinfo.additional_flags << " "
                << " -sd " << compileinfo.net_info.out_path
                << " -sp " << compileinfo.net_info.prefix
                << " -f " << compileinfo.fmem_bank_num << " " << compileinfo.fmem_bank_size
                << " -w " << compileinfo.cim_num << " " << compileinfo.wmem_size << " " << compileinfo.ewmem_size
                << " -ps " << compileinfo.packet_size
                << " -mo " << compileinfo.net_info.meminfo.shared_offset
                << " " << compileinfo.net_info.meminfo.input_offset
                << " " << compileinfo.net_info.meminfo.output_offset
                << " " << compileinfo.net_info.meminfo.wb_offset
                << " " << compileinfo.net_info.meminfo.buf_offset;

        std::cout << command.str() << "\n";

        int ret = system(command.str().c_str());
        if (ret < 0) {
            std::cout << "Failed to execute a MIDAP compilation.\n";
            throw std::runtime_error("MIDAP compilation failed.");
        }

        compileinfo.net_info.precompiled = true;
    }
}

void Configurations::init_sync_wait_map()
{
    NetworkInfo netinfo = get_netinfo();
    std::string base_path = netinfo.out_path + netinfo.prefix + "/core_1/";
    static const std::array<std::pair<std::string, SyncMap&>, 2> sync_maps = {{
        {"cpu", host_sync_map},
        {"pim", pim_sync_map}
    }};

    static const std::array<std::pair<std::string, WaitMap&>, 2> wait_maps = {{
        {"cpu", host_wait_map},
        {"pim", pim_wait_map}
    }};

    for (const auto& [device_name, sync_map] : sync_maps) {
        load_sync_wait_info(base_path + device_name + "_sync_info.txt", sync_map);
    }
    
    for (const auto& [device_name, wait_map] : wait_maps) {
        load_sync_wait_info(base_path + device_name + "_wait_info.txt", wait_map);
    }
    
    std::cout << "[HOST_SYNC_MAP]:\n";
    for (const auto& [key, value] : host_sync_map) {
        unsigned int id, size;
        uint64_t address;
        
        std::tie(id, size, address) = value;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size(B): " << size
                  << " | Address: " << address << "\n";
    }
    std::cout << "[HOST_WAIT_MAP]:\n";
    for (const auto& [key, value] : host_wait_map) {
        unsigned int id, size;
        uint64_t address;
        
        std::tie(id, size, address) = value;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size(B): " << size
                  << " | Address: " << address << "\n";
    }
    std::cout << "[PIM_SYNC_MAP]:\n";
    for (const auto& [key, value] : pim_sync_map) {
        unsigned int id, size;
        uint64_t address;
        
        std::tie(id, size, address) = value;
        pim_sync_id_map[id] = true;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size: " << size
                  << " | Address: " <<  address << "\n";
    }
    std::cout << "[PIM_WAIT_MAP]:\n";
    for (const auto& [key, value] : pim_wait_map) {
        unsigned int id, size;
        uint64_t address;
        
        std::tie(id, size, address) = value;
        pim_wait_id_map[id] = true;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size: " << size
                  << " | Address: " << address << "\n";
    }
}

void Configurations::load_sync_wait_info(const std::string& file_path, BaseMap& info_map) {
    std::ifstream file(file_path);
    if (!file) {
        std::cerr << "Error: Cannot open file " << file_path << "\n";
        return;
    }

    std::string line;

    while (getline(file, line)) {
        std::istringstream ss(line);
        std::string layer;
        unsigned int id, dim;
        uint64_t mem_id, offset;

        if (!(ss >> id) || ss.get() != ',' ||
            !(ss >> mem_id) || ss.get() != ',' ||
            !(ss >> offset) || ss.get() != ',' ||
            !(ss >> dim) || ss.get() != ',' ||
            !getline(ss, layer)) {
            std::cerr << "Warning: Invalid line format -> " << line << "\n";
            continue;
        }

        uint64_t base_address = (mem_id == 1)
                                ? compileinfo.net_info.meminfo.input_offset
                                : compileinfo.net_info.meminfo.output_offset;
                                
        info_map.insert({layer, {id, dim*2, base_address + offset}}); // FP16(2B)
    }
}

Json::Value Configurations::parse_json(const std::string& file_name)
{
    std::string filePath = std::string(ROOT_PATH) + "hsim/configs/" + file_name + ".json";

    std::ifstream ifs(filePath);
    if(!ifs) {
        std::cerr << filePath << " file doesn't exist.\n";
        exit(EXIT_FAILURE);
    }

    std::string rawJson;

    ifs.seekg(0, std::ios::end);
    rawJson.reserve(ifs.tellg());
    ifs.seekg(0, std::ios::beg);

    rawJson.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

    JSONCPP_STRING err;
    Json::Value root;
    Json::CharReaderBuilder builder;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    reader->parse(rawJson.c_str(), rawJson.c_str() + rawJson.length(), &root, &err);

    return root;
}