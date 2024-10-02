import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import torch
import torch.nn as nn
import datetime

now = datetime.datetime.now()
now = now.strftime('%m-%d-%H.%M')



class LSTMModel(nn.Module):
    def __init__(self,max_len, vocab_size=5, emb_dim=100, hidden_dim=64):
        super().__init__()

        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        layer_num = 2
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=layer_num)
        self.h0 = torch.randn(layer_num, self.max_len, hidden_dim)  # 初始隐藏状态，层数为2，批次大小为3，隐藏维度为20
        self.c0 = torch.randn(layer_num, self.max_len, hidden_dim)  # 初始细胞状态
        self.dropout = nn.Dropout(p=0.3)

        self.block = nn.Sequential(
            nn.Linear(64 * max_len, 1024),  # 415:26560 201:12864 101:6464 51:3264
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )


    def forward(self, x):
        x = x.to('cuda:3')
        x = self.embedding(x)  # torch.Size([32, 415, 100])
        x = self.dropout(x)
        self.h0 = self.h0.to('cuda:3')
        self.c0 = self.c0.to('cuda:3')
        output, (hn, cn) = self.lstm(x, (self.h0, self.c0))  # 输出[4019, 52, 64]

        output = output.reshape(output.shape[0], -1)  # [4019, 3328]
        #  print(output.shape,hn.shape)
        output = self.block(output)
        # x = self.fc(output)
        return output, 1
