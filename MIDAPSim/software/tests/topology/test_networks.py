import shutil

import pytest

import models.inception as inception
import models.mobilenet as mobilenet
import models.resnet as resnet
import models.se_resnet as se_resnet
from config import cfg
from software.tests.net_test_utils import (
    is_dot_files_in_dirs_same,
    draw_graph,
    set_and_get_tmp_dot_directory,
)

custom_examples = {
    "resnet50": (
        lambda x: resnet.resnet50(input_size=x[-1])
        if x is not None
        else resnet.resnet50()
    ),
    "inception_v3": (
        lambda x: inception.inception_v3(input_size=x[-1])
        if x is not None
        else inception.inception_v3()
    ),
    "se_resnet50": (
        lambda x: se_resnet.se_resnet50(input_size=x[-1])
        if x is not None
        else se_resnet.se_resnet50()
    ),
    "mobilenet_v2": (
        lambda x: mobilenet.mobilenet_v2(input_size=x[-1])
        if x is not None
        else mobilenet.mobilenet_v2()
    ),
}

cfg.SYSTEM.BANDWIDTH = (
    cfg.DRAM.CHANNEL_SIZE * cfg.DRAM.FREQUENCY * cfg.DRAM.NUM_CHANNELS * 2
) // cfg.SYSTEM.DATA_SIZE


@pytest.mark.parametrize("network_name", custom_examples.keys())
def test_all_cases(network_name):
    correct_dot_folder_name = "software/tests/golden"
    temp_folder_name = set_and_get_tmp_dot_directory(suffix=network_name)

    shape = [1, 3, 224, 224] if network_name != "inception_v3" else [1, 3, 299, 299]

    mb = custom_examples[network_name](shape)
    draw_graph(mb, model_name=network_name)

    print(f"{correct_dot_folder_name} {temp_folder_name}")
    is_same = is_dot_files_in_dirs_same(
        correct_dot_folder_name, temp_folder_name, dotfile_name=network_name
    )
    shutil.rmtree(temp_folder_name)
    assert is_same
