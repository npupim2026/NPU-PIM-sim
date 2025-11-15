import pytest
from config import cfg
from test_wrapper import TestWrapper
from models.custom_torch.examples.densenet import get_densenet

cfg.SYSTEM.BANDWIDTH = (cfg.DRAM.CHANNEL_SIZE * cfg.DRAM.FREQUENCY * cfg.DRAM.NUM_CHANNELS * 2) // cfg.SYSTEM.DATA_SIZE

expected_MAC = {
    'densenet121': 2834161664,
    'densenet161': 7727907072,
    'densenet169': 3359843328,
    'densenet201': 4291365888,

}


@pytest.mark.parametrize("model_name", list(expected_MAC.keys()))
def test_densenet(model_name):
    input_shape = (1, 3, 224, 224)

    blocks = int(model_name[-3:])
    mb = get_densenet(blocks=blocks, model_name=model_name)

    tr = TestWrapper(simulation_level=2)
    _, _ = tr.run_all(mb)
    assert tr.midap_simulator.stats.global_stats.MACs == expected_MAC[model_name]

if __name__=='__main__':
    test_densenet('densenet121')