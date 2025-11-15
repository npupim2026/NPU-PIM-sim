import os
from software.compiler.compile_info import CompileInfo
import time

from config import cfg
from software.compiler import Compiler
from software.generic_op import GenericModel
from software.network import ModelGraph


def set_and_get_tmp_dot_directory(suffix=""):
    temp_folder_name = "tmp" + str(time.time()) + suffix
    if os.path.exists(temp_folder_name):
        raise OSError("temp directory exists")
    os.makedirs(temp_folder_name)

    cfg.MODEL.DOT_DIRECTORY = temp_folder_name
    cfg.MODEL.GENERATE_MODEL_DOT = True
    cfg.MODEL.GENERATE_BLOCK_DOT = False

    return temp_folder_name


def is_dot_files_in_dirs_same(dir1, dir2, dotfile_name):
    correct_dot_path = os.path.join(dir1, dotfile_name + ".gv")
    compiled_dot_path = os.path.join(dir2, dotfile_name + ".gv")
    print(f"Correct {correct_dot_path} {compiled_dot_path}")
    is_same = is_dots_same(correct_dot_path, compiled_dot_path)
    return is_same


def is_dots_same(dot1, dot2):
    g1 = open(dot1, "r").readlines()
    g2 = open(dot2, "r").readlines()
    return is_digraphs_same(g1, g2)


def is_digraphs_same(g1: list, g2: list):
    e1 = make_edge_set(g1)
    e2 = make_edge_set(g2)
    return e1 == e2


def make_edge_set(g1: list):
    result = set()
    for s in g1:
        assert isinstance(s, str)
        if is_edge(s):
            result.add(s)

    return result


def is_edge(s: str):
    return "->" in s


def draw_graph(mb, model_name):
    odict = mb.get_operator_dict()

    # Convertor
    cv = GenericModel()
    cv.operator_dict = odict
    cv.post_process()

    # Model
    model = ModelGraph(model_name)
    model.build(odict)
    if cfg.MODEL.GENERATE_MODEL_DOT:
        model.draw(cfg.MODEL.DOT_DIRECTORY)
