import torch

from .dyhead import DyHead
from .dyrelu import DYReLU
from .dropblock import DropBlock2D


class Scale(torch.nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = torch.nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
