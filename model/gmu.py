# Created by ttwu at 2022/8/14
import copy

import torch.nn as nn
import torch
from collections import namedtuple
from transformers import ViltConfig, PreTrainedTokenizer, PreTrainedModel
from transformers.modeling_utils import *

from module.modeling_vilt import ViltModel
from module.head import ClassificationHead
from module.encoders import TextEncoder, ImageEncoder
from utils import signature
from configs.config_base import MmreConfigBase
from torch.nn import init
import torch.nn.functional as F

from torch.nn.parameter import Parameter
import math
import pdb
class GatedBimodal(nn.Module):
    def __init__(self, dim, activation=None,gate_activation=None):
        super(GatedBimodal, self).__init__()
        self.dim=dim
        if not activation:
            activation = nn.Tanh()
        if not gate_activation:
            gate_activation = nn.Sigmoid()
        self.activation = activation
        self.gate_activation = gate_activation
        self.W = nn.Parameter(torch.Tensor(2*self.dim, self.dim),requires_grad=True)
        nn.init.kaiming_normal(self.W, mode='fan_out')
        # sampling
    def forward(self, x1,x2):
        x=torch.cat([x1,x2],dim=1)
        h=self.activation(x)
        z=self.gate_activation(torch.matmul(x,self.W))
        # TODO: split text and image embedding from embedding_output

        return z * h[:, :self.dim] + (1 - z) * h[:, self.dim:], z

class GatedClassifier(nn.Module):
    def __init__(self, visual_dim, text_dim,output_dim,hidden_size):
        super(GatedClassifier, self).__init__()
        self.vismlp=nn.Sequential(nn.BatchNorm1d(visual_dim),
        nn.Linear(visual_dim,hidden_size))
        self.textmlp=nn.Sequential(nn.BatchNorm1d(text_dim),
        nn.Linear(text_dim,hidden_size))
        # self.vismlp=nn.Linear(visual_dim,hidden_size)
        # self.textmlp=nn.Linear(text_dim,hidden_size)
        self.gbu=GatedBimodal(hidden_size)
        self.classifier=MLPGenreClassifier(hidden_size,output_dim,hidden_size)
        # sampling
    def forward(self, x1,x2):
        vh=self.vismlp(x1)
        th=self.textmlp(x2)
        h,z=self.gbu(vh,th)
        y_hat=self.classifier(h)
        
        # TODO: split text and image embedding from embedding_output

        return y_hat,z





class Maxout(nn.Module):
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, pieces, bias=True):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pieces = pieces
        self.weight = Parameter(torch.Tensor(pieces, out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(pieces, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = input.matmul(self.weight.permute(0, 2, 1)).permute((1, 0, 2)) + self.bias
        output = torch.max(output, dim=1)[0]
        return output

class MLPGenreClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(MLPGenreClassifier, self).__init__()
        # self.linear1=Maxout(input_dim,hidden_size,2)
        # self.linear2=Maxout(hidden_size,hidden_size,2)
        self.linear1=nn.Linear(input_dim,hidden_size)
        self.linear2=nn.Linear(hidden_size,hidden_size)

        self.linear3=nn.Linear(hidden_size,output_dim)
        # sampling
    def forward(self, x):


        return torch.sigmoid(self.linear3(self.linear2(self.linear1(x))))
