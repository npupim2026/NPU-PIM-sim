#ifndef TOP_H
#define TOP_H

#include "host.h"
#include "midap_wrapper.h"
#include "mem_wrapper.h"
#include "pim_wrapper.h"
#include "arbiter.h"

unsigned int active_host;
unsigned int active_core;
unsigned int active_dram;
extern Configurations cfgs;

SyncObject dram_sync_obj;
SyncObject pim_sync_obj;
SyncObject pim_npu_sync_obj;

SC_MODULE(Top)
{
    std::unique_ptr<Host> host;
    std::unique_ptr<Arbiter> arbiter;
    std::unique_ptr<MIDAPWrapper> midap_wrapper;
    std::unique_ptr<MEMWrapper> mem_wrapper;
    std::unique_ptr<PIMWrapper> pim_wrapper;
    sc_in<bool> clock;
    Logger logger;

    SC_CTOR(Top)
    {
        instantiate_modules();
        bind_modules();
        bind_clocks();

        active_host = 1;
        active_core = 1;
        active_dram = 1;
    }

private:
    void instantiate_modules() {
        host = std::make_unique<Host>("host", &logger);
        midap_wrapper = std::make_unique<MIDAPWrapper>("midap_wrapper", &logger);
        arbiter = std::make_unique<Arbiter>("arbiter");

        if (cfgs.dram_enabled()) { mem_wrapper = std::make_unique<MEMWrapper>("mem_wrapper", &logger); }
        if (cfgs.pim_enabled())  { pim_wrapper = std::make_unique<PIMWrapper>("pim_wrapper", &logger); }
    }

    void bind_modules() {
        host->master.bind(midap_wrapper->slave);
        host->master.bind(arbiter->slave);
        midap_wrapper->master.bind(arbiter->slave);

        if (mem_wrapper) { arbiter->master.bind(mem_wrapper->slave); }
        if (pim_wrapper) { arbiter->master.bind(pim_wrapper->slave); }
    }

    void bind_clocks() {
        host->clock(clock);
        arbiter->clock(clock);
        midap_wrapper->clock(clock);

        if (mem_wrapper) { mem_wrapper->clock(clock); }
        if (pim_wrapper) { pim_wrapper->clock(clock); }
    }
};

#endif //TOP_H