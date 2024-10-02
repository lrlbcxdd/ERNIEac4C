import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class model(nn.Module):
    def __init__(self,max_len):
        super(model,self).__init__()
        self.max_len = max_len
        self.hidden_dim = 25
        self.batch_size = 64
        self.emb_dim = 256

        self.embedding = nn.Embedding(5,self.emb_dim,padding_idx=0)
        self.GRU = nn.GRU(self.emb_dim,self.hidden_dim,num_layers=2,bidirectional=True,dropout=0.2)

        # 用GRU的线性层
        self.block = nn.Sequential(nn.Linear(self.hidden_dim * (2 * self.max_len + 4), 1024),
                                   nn.BatchNorm1d(1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 256),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(),
                                   nn.Linear(256,2)
                                   )


    def forward(self,x):
        x = self.embedding(x).permute(1,0,2)       # torch.Size([128, max_len, 512])

        # 用GRU
        output , hn = self.GRU(x)  # output: torch.Size([max_len, 128, hidden_dim*2]) hn: torch.Size([4, 128, 25])
        output = output.permute(1,0,2) # output: torch.Size([128, max_len, hidden_dim*2])
        hn = hn.permute(1,0,2)  # torch.Size([128, 4, hidden_dim])

        output = output.reshape(output.shape[0], -1) # torch.Size([128, max_len * hidden_dim *2])
        hn = hn.reshape(output.shape[0],-1) # torch.Size([128, 4* hidden_dim])
        output = torch.cat([output, hn], 1)   # output:torch.Size([128, hidden_dim *(2*max_len + 4)])

        output = self.block(output)  # output:torch.Size([128, 1024])

        return output , 1
