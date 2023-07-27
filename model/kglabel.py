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
        self.W1 = nn.Parameter(torch.Tensor(self.dim, self.dim),requires_grad=True)
        self.W2 = nn.Parameter(torch.Tensor(self.dim, self.dim),requires_grad=True)
        self.W3 = nn.Parameter(torch.Tensor(self.dim, self.dim),requires_grad=True)

        nn.init.kaiming_normal(self.W1, mode='fan_out')
        nn.init.kaiming_normal(self.W2, mode='fan_out')
        nn.init.kaiming_normal(self.W3, mode='fan_out')
        # sampling
    def forward(self, x1,x2,x3):
        # x=torch.cat([x1,x2,x3],dim=1)
        h1=self.activation(x1)
        h2=self.activation(x2)
        h3=self.activation(x3)
        z1=self.gate_activation(torch.matmul(x1,self.W1))
        z2=self.gate_activation(torch.matmul(x2,self.W2))
        z3=self.gate_activation(torch.matmul(x3,self.W3))

        # z=self.gate_activation(torch.matmul(x,self.W))
        # TODO: split text and image embedding from embedding_output

        return z1*h1+z2*h2 +z3*h3 , z1+z2+z3

class GatedClassifier(nn.Module):
    def __init__(self, visual_dim, text_dim,kg_dim,output_dim,hidden_size):
        super(GatedClassifier, self).__init__()
        self.vismlp=nn.Sequential(nn.BatchNorm1d(visual_dim),
        nn.Linear(visual_dim,hidden_size))
        self.textmlp=nn.Sequential(nn.BatchNorm1d(text_dim),
        nn.Linear(text_dim,hidden_size))
        self.kgmlp=nn.Sequential(nn.BatchNorm1d(kg_dim),
        nn.Linear(kg_dim,hidden_size))
        # self.labelmlp=nn.Sequential(nn.BatchNorm1d(kg_dim),
        # nn.Linear(kg_dim,hidden_size))
        # self.vismlp=nn.Linear(visual_dim,hidden_size)
        # self.textmlp=nn.Linear(text_dim,hidden_size)
        self.gbu=GatedBimodal(hidden_size)
        self.classifier=MLPGenreClassifier(hidden_size,output_dim,hidden_size)
        self.labelaw=labelaware()
        # sampling
    def forward(self, x1,x2,x3,x4):
        vh=self.vismlp(x1)
        th=self.textmlp(x2)
        kh=self.kgmlp(x3)
        h,z=self.gbu(vh,th,kh)
        labaw=self.labelaw(x4)
        y_hat=self.classifier(h,labaw)
        
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
    def forward(self, x,y):


        return torch.sigmoid(y*self.linear3(self.linear2(self.linear1(x))))


class labelaware(nn.Module):
    
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool1d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()
        self.max=nn.AdaptiveMaxPool1d(1)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).squeeze(-1) #bs,c
        ##
        # y=x*y.expand_as(x)
        # y=self.max(y).squeeze(-1)
        return y