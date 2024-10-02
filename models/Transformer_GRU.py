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
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim,nhead=8)
        self.transformer_encoder_seq = nn.TransformerEncoder(self.encoder_layer,num_layers=1)
        self.GRU = nn.GRU(self.emb_dim,self.hidden_dim,num_layers=2,bidirectional=True,dropout=0.2)


        # 用GRU的线性层
        self.block = nn.Sequential(nn.Linear(self.hidden_dim *(2 * self.max_len + 4),2048),
                                   nn.BatchNorm1d(2048),
                                   nn.LeakyReLU(),
                                   nn.Linear(2048,1024),
                                   )

        # 不用GRU的线性层
        # self.block = nn.Sequential(nn.Linear(self.emb_dim * max_len,2048),
        #                            nn.Dropout(0.25),
        #                            nn.ReLU(),
        #                            nn.Linear(2048,1024),
        #                            )

        self.block1 = nn.Sequential(nn.Linear(1024, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256)
                                    )

        self.block2 = nn.Sequential(nn.Linear(256, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64))

        self.block3 = nn.Sequential(nn.Linear(64, 8),
                                    nn.Dropout(0.25),
                                    nn.ReLU(),
                                    nn.Linear(8, 2))

    def forward(self,x):
        x = self.embedding(x)       # torch.Size([128, max_len, 512])

        # 用GRU
        output = self.transformer_encoder_seq(x).permute(1,0,2) # torch.Size([max_len, 128, 512])
        output , hn = self.GRU(output)  # output: torch.Size([max_len, 128, hidden_dim*2]) hn: torch.Size([4, 128, 25])
        output = output.permute(1,0,2) # output: torch.Size([128, max_len, hidden_dim*2])
        hn = hn.permute(1,0,2)  # torch.Size([128, 4, hidden_dim])

        output = output.reshape(output.shape[0], -1)  # torch.Size([128, max_len * hidden_dim *2])
        hn = hn.reshape(output.shape[0], -1)  # torch.Size([128, 4* hidden_dim])
        output = torch.cat([output, hn], 1)   # output:torch.Size([128, hidden_dim *(2*max_len + 4)])

        # output = output[:, 0, :]  # torch.Size([64, 512])
        # 不用GRU
        # output = self.transformer_encoder_seq(x) # torch.Size([128, max_len , 512])
        # output = output.reshape(output.shape[0], -1) # torch.Size([128, max_len * hidden_dim *2])

        output = self.block(output) # output:torch.Size([128, 1024])
        x = self.block1(output) # output:torch.Size([128, 256])
        x = self.block2(x)  # output:torch.Size([128, 64])
        output = self.block3(x) # x:torch.Size([128, 64]) # output:torch.Size([128, 2])

        return output , x
