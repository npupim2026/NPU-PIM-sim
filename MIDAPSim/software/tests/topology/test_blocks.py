import shutil

import pytest

from models.json_model.blockargs import BlockArgs, StrBlockArgsDecoder
from models.json_model.model import OnlyBlockModel
from models.json_model.utils import TupleStyleDict, get_kwargs
from software.tests.net_test_utils import (
    is_dot_files_in_dirs_same,
    draw_graph,
    set_and_get_tmp_dot_directory,
)


@pytest.mark.parametrize("has_skip_con", [True, False])
@pytest.mark.parametrize("se_ratio", [0, 0.25])
def test_mixconv_block(has_skip_con, se_ratio):
    kwargs = get_kwargs()
    block_test_template(
        block_type="MixConvBlock", expand_ratio=[2, 2], kernel_size=[3, 3], **kwargs
    )


@pytest.mark.parametrize("block_type", ["MBConvBlock", "ConvBlock_kxk1x1"])
@pytest.mark.parametrize("has_skip_con", [True, False])
@pytest.mark.parametrize("se_ratio", [0, 0.25])
def test_other_block(block_type, has_skip_con, se_ratio):
    kwargs = get_kwargs()
    block_test_template(expand_ratio=4, kernel_size=3, **kwargs)


def block_test_template(block_type, expand_ratio, kernel_size, has_skip_con, se_ratio):
    block_args = get_block_args(
        conv_type=block_type,
        expand_ratio=expand_ratio,
        kernel_size=kernel_size,
        id_skip=has_skip_con,
        se_ratio=se_ratio,
    )

    block_str = StrBlockArgsDecoder._encode_block_string(block_args)
    temp_folder_name = set_and_get_tmp_dot_directory(suffix=block_str)

    input_size = 7
    mb = make_modelbuilder_with_1blockargs(block_args, input_size)
    draw_graph(mb, model_name=block_str)

    correct_dot_folder_name = "software/tests/golden"
    is_same = is_dot_files_in_dirs_same(
        correct_dot_folder_name, temp_folder_name, dotfile_name=block_str
    )
    shutil.rmtree(temp_folder_name)
    assert is_same


def make_modelbuilder_with_1blockargs(block_args, input_size):
    model_json = get_default_model_json(
        first_conv_output_filters=block_args.input_filters, first_stride=1
    )
    model_json["stages_args"] = [{"cls": "BasicStage", "blocks_args": [block_args]}]

    input_shape = (1, 3, input_size, input_size)
    mb = OnlyBlockModel()(model_json=model_json, input_shape=input_shape)
    return mb


def get_default_model_json(
    stem3x3=False,
    first_conv_output_filters=32,
    first_stride=2,
    last_conv_output_filters=1280,
):
    return {
        "img_size": 224,
        "img_c": 3,
        "n_classes": 1000,
        "first_conv": {
            "cls": "Conv2D",
            "output_filters": first_conv_output_filters,
            "kernel_size": 7 if not stem3x3 else 3,
            "strides": first_stride,
        },
        "feature_mix_layer": {
            "cls": "Conv2D",
            "output_filters": last_conv_output_filters,
            "kernel_size": 1,
        },
    }


def get_block_args(
    num_repeat=1,
    conv_type="MBConvBlock",
    kernel_size=3,
    input_filters=32,
    output_filters=32,
    expand_ratio=2,
    strides=[1, 1],
    id_skip=True,
    se_ratio=0.25,
    act_fn=None,
):
    kwargs = get_kwargs()
    return TupleStyleDict(BlockArgs(**kwargs)._asdict())
