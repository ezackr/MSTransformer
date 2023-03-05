from torch import nn

from mstransformer.src.transformer.mstu import MSTU
from mstransformer.src.utils.analysis import get_number_of_parameters


def test_get_number_of_parameters_linear():
    model = nn.Linear(32, 10)
    assert get_number_of_parameters(model) == 330


def test_get_number_of_parameters_mstu():
    model = MSTU()
    assert get_number_of_parameters(model) == 17561216
