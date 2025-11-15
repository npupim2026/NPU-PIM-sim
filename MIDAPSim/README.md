G# A Novel CNN Accelerator That Enables Fully-Pipelined Execution of Layers

By [Donghyun Kang](http://iris.snu.ac.kr/xe/kangdongh), [Jintaek Kang](http://iris.snu.ac.kr/xe/taek0208), [Soonhoi Ha](https://peace.snu.ac.kr/sha/).

# Introduction

MIDAP, Memory In the Datapath Architecture Processor, features bus-free multi-bank on-chip memory architecture. For more details, please refer to our [ICCD Paper](https://ieeexplore.ieee.org/document/8988663).

# Citing MIDAP

Please cite MIDAP in your publications if it helps your research:

    @inproceedings{kang2019novel,
        title = {A Novel Convolutional Neural Network Accelerator That Enables Fully-Pipelined Execution of Layers},
        author = { D. {Kang} and J. {Kang} and H. {Kwon} and H. {Park} and S. {Ha} },
        booktitle = { 2019 IEEE 37th International Conference on Computer Design (ICCD) },
        year = {2019},
        pages = {698--701},
    }

This repository includes MIDAP Compiler & MIDAP Simulator

* Midap Simulator can be excuted with dedicated simulator instruction, please see data_structure/simulator_instruction.py

# Tensor Virtualization

1. We have proposed a novel virtualization technique and tested it via MIDAPSim.

    @inproceedings{kang2020tensor,
        title={Tensor virtualization technique to support efficient data reorganization for CNN accelerators},
        author={Kang, Donghyun and Ha, Soonhoi},
        booktitle={2020 57th ACM/IEEE Design Automation Conference (DAC)},
        pages={1--6},
        year={2020},
        organization={IEEE}
    }

2. Tensor virtualization technique is applied to Concat, Upsample (NN, Bilinear), TransposedConv (UpsampleZero + Conv) Layers.

# How to download & simulate?

``` Shell
user@123:~/# git clone $REPO_URL $MIDAPSIM_ROOT ; cd $MIDAPSIM_ROOT
user@123:$MIDAPSIM_ROOT# git submodule init
user@123:$MIDAPSIM_ROOT# git submodule update
user@123:$MIDAPSIM_ROOT# 
    python tools/test_system.py [-h] -n
        {TARGET_NETWORK: dcgan,discogan,unet,resnet50,resnet101,se_resnet50,se_resnet101,mobilenet,mobilenet_v2,mobilenet_v3_small, ...}
        [-i INPUT_SIZE] [-W SYSTEM_WIDTH] [-f NUM_FMEM FMEM_SIZE] [-w NUM_WMEM WMEM_SIZE E_WMEM_SIZE] [--level LEVEL] [--debug] [-d {DMA, VIRTUAL}]
        [-gt GRAPH_TRANSFORMER] [-mp MAPPING_POLICY] [-lmc LOCAL_MEMORY_COMPILER] [-lsc L2_SPM_COMPILER]
        [--spm_config SPM_CONFIG SPM_CONFIG] [-nc NUM_CORES] [-fs] [-q] [-so] [-sd SAVE_DIR] [-sp SAVE_PREFIX]
```

# Save Compile Result with quantized model & Run it independent from compiler!

1. Compile & Save the result: Use these options

* -so: save only
* -sd: save diredtory
* -sp: save prefix

* -q: make model quantized
* -fs: activate functional simulation

2. Directory $SAVE\_ROOT = ($SAVE\_DIR)/($SAVE\_PREFIX) will be generated if you saved the compile result.

* You can modify $SAVE\_ROOT/$(CORE\_ID)/config.yml file to modify your configuration .. but it is not recommended.

3. Run from saved file
``` Python
python run_from_file.py -d $DIRECTORY -p $PREFIX [-id $CORE_ID] [-l $SIM_LEVEL] [--debug]
```
* these options are used for non-functional simulation or system-wise simulation

# Patch Notes for MIDAP System Compiler branch

## v1.0.0

1. System Compilation Support

2. Refined Compiler Data Structure
