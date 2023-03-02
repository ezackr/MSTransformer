import torch

from mstransformer.src.transformer.sampling import ConvLayer, DownSample, UpSample


def test_convlayer():
    x_in = torch.rand(size=(32, 2, 44100))
    conv = ConvLayer(2, 4)
    x_out = conv(x_in)
    print(x_out.shape)



