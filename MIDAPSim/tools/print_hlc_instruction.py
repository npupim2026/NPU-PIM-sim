import os
import sys
sys.path.append(os.path.dirname("./"))
import pickle
import argparse
from data_structure.attrdict import AttrDict, from_dict_to_attrdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inst_file', metavar='path', type=str, help='The path to the instruction file')
    inst_file = parser.parse_args().inst_file
    with open(inst_file, 'rb') as f:
        ins = pickle.load(f)
    print('<<<Address Information>>>')
    for item in ins['addr_dict'].items():
        print('\t' + str(item))
    print()
    print('<<<Layer Information>>>')
    for item in ins['processing_order']:
        print(str(item) + '\n')
