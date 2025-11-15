from .base import ControlLogic
from .level0 import ControlLogicLv0, ControlLogicLv1
from .level2 import ControlLogicLv2

def get_control_logic(manager, simulation_level):
    if simulation_level == -1:
        return ControlLogic(manager)
    if simulation_level == 0:
        return ControlLogicLv0(manager)
    if simulation_level == 1:
        return ControlLogicLv1(manager)
    if simulation_level == 2:
        return ControlLogicLv2(manager)
    # if simulation_level == 3:
    #     return ControlLogicLv3(manager)

