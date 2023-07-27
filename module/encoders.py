import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()

    def forward(self, x):
        return 0


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()

    def forward(self, x):
        return 1

