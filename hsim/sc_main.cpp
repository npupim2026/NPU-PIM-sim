#include <components/Top.h>

Configurations cfgs;
SyncObject sync_object;

int sc_main(int argc, char* argv[])
{
    double host_freq, mcu_freq;

    cfgs.init_configurations();

    Top top("top");

    host_freq = cfgs.get_host_freq(); /* GHz */

    sc_clock clk("clk", sc_time(1/host_freq, SC_NS));
    top.clock(clk);

    clock_t start_clock = clock();
    sc_start();
    clock_t end_clock = clock();

    cfgs.print_configurations();

    cout << "Simulation Time : " << std::setprecision(5) << (double) (end_clock - start_clock) / CLOCKS_PER_SEC << " (seconds)\n";
    cout << "Simulated Cycle : " << (int) (sc_time_stamp().to_double() / 1000 / 1/host_freq) << " (cycles)\n";
    cout << "Simulated Time  : " << (int) (sc_time_stamp().to_double() / 1000) << " (ns)\n";
    return 0;
}
