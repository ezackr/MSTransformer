import torch
from torch import nn

from mstransformer.src.transformer.mstu import MSTU
from mstransformer.src.utils.analysis import sdr, get_number_of_parameters


def test_get_number_of_parameters_linear():
    model = nn.Linear(32, 10)
    assert get_number_of_parameters(model) == 330


def test_get_number_of_parameters_mstu():
    model = MSTU()
    assert get_number_of_parameters(model) == 17561216


def test_sdr():
    estimate = torch.ones(size=(32, 2, 2000))
    target = 1000 * torch.ones(size=(32, 2, 2000))
    score = sdr(target, estimate)
    assert score <= 0.010
    assert score > 0
