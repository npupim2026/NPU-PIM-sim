cmake_minimum_required(VERSION 3.14)
project(hsim)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DRAMULATOR_DEBUG")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(.
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/systemc-3.0.0/include
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/spdlog/include
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/jsoncpp/include
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}/../ramulator2/ext/yaml-cpp/include
	${CMAKE_CURRENT_SOURCE_DIR}/../ramulator2/ext/spdlog/include
	${CMAKE_CURRENT_SOURCE_DIR}/../ramulator2/ext/argparse/include
	${CMAKE_CURRENT_SOURCE_DIR}/../ramulator2/src
	${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3/src
	${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3/ext/headers
	)

link_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/systemc-3.0.0/lib-linux64
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/spdlog
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/jsoncpp
	${CMAKE_CURRENT_SOURCE_DIR}/../ramulator2
	${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3
	)

add_definitions(
	-DROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../"
	-DDRAM_RAMULATOR
	-DDRAM_DRAMSIM3
	-DRAMULATOR_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../ramulator2/"
	-DDRAMSIM3_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3/"
	)

link_libraries(rt systemc-3.0.0 spdlog jsoncpp ramulator dramsim3)

add_executable(hsim
	sc_main.cpp
	components/host.cpp
	components/arbiter.cpp
	components/midap_wrapper.cpp
	components/mem_wrapper.cpp
	components/pim_wrapper.cpp
	components/mcu.cpp
	components/trace_generator.cpp
	utilities/mm.cpp
	utilities/logger.cpp
	utilities/packet_buffer.cpp
	utilities/shmem_communicator.cpp
	utilities/configurations.cpp
	scenarios/baseline.cpp
	scenarios/unified.cpp
	scenarios/partitioned.cpp
	)
