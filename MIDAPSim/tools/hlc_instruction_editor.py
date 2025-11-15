import pickle
import code
import argparse
from copy import deepcopy
import os
import sys
sys.path.append(os.path.dirname("./"))
from data_structure.instruction_components import SLayerInfo


class HLCInfo:
    def __init__(self, path: str):
        self._path: str = path
        self._content = {}

    def load(self):
        with open(self._path, 'rb') as f:
            self._content = pickle.load(f)

    def save(self):
        with open(self._path, 'wb') as f:
            pickle.dump(self._content, f)


class HLCInst(HLCInfo):
    @property
    def addr(self):
        return self._content['addr_dict']

    @property
    def inst(self):
        return self._content['processing_order']

    def _find_layer_by_name(self, name: str):
        try:
            return next(filter(lambda l: l.name == name, self.inst))
        except StopIteration:
            return None

    def add_layer(self, other: HLCInfo, name: str):
        layer = other._find_layer_by_name(name)
        if layer is not None:
            self.inst.append(deepcopy(layer))

    #def add_layer(self, layer: SLayerInfo):
    #    self.inst.append(deepcopy(layer))

    #def replace_layer(self, other: HLCInfo, name: str):
    #    pass


class HLCData(HLCInfo):
    @property
    def data(self):
        return self._content[0]

    def add_layer(self, other: HLCInfo, name: str):
        self.data[name] = other.data[name]


class HLCCoreInfo:
    def __init__(self, dir_path: str):
        self._dir_path = dir_path
        self.inst = HLCInst(dir_path + '/inst.pkl')
        self.data = HLCData(dir_path + '/gt_info.pkl')

    def load(self):
        self.inst.load()
        self.data.load()

    def save(self):
        self.inst.save()
        self.data.save()

    def add_layer(self, other, name: str):
        self.inst.add_layer(other.inst, name)
        self.data.add_layer(other.data, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path', metavar='path', type=str, help='The path to the directory where inst.pkl and gt_info.pkl located')
    path = parser.parse_args().dir_path
    info = HLCCoreInfo(dir_path=path)
    info.load()
    code.interact(local=locals())
