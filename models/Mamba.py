import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

torch.autograd.set_detect_anomaly(True)


# Configuration flags and hyperparameters
USE_MAMBA = 1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0

device = torch.device("cuda", 0)
batch_size = 64  # Example batch size
different_batch_size = False
h_new = None
temp_buffer = None


# 通过一系列线性变换和离散化过程处理输入序列
class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size):
        super(S6, self).__init__()

        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, state_size)
        self.fc3 = nn.Linear(d_model, state_size)

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size)

        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size)

        # h [batch_size, seq_len, d_model, state_size]
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model)

    def discretization(self):

        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))

        return self.dA, self.dB

    def forward(self, x):
        # Algorithm 2 MAMBA paper
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        self.discretization()

        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:

            global current_batch_size
            current_batch_size = x.shape[0]

            if self.h.shape[0] != current_batch_size:
                different_batch_size = True
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x,
                                                                                                               "b l d -> b l d 1") * self.dB

            else:
                different_batch_size = False
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y [batch_size, seq_len, d_model]
            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()

            return self.y

        else:
            # h [batch_size, seq_len, d_model, state_size]
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size , device=device )
            y = torch.zeros_like(x)
            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB
            # y [batch_size, seq_len, d_model]
            y = torch.einsum('bln,bldn->bld', self.C, h)

            return y


class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size):
        super(Mamba, self).__init__()

        self.embedding = nn.Embedding(5,d_model)

        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size)

        self.fc_task = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model // 2),
        )
        self.classifier = nn.Linear(d_model // 2, 2)

    def forward(self, x):
        x = self.embedding(x)   # torch.Size([b, l, d_model])

        x = self.mamba_block1(x)
        # x = self.mamba_block2(x)
        # x = self.mamba_block3(x)

        # output_encoder = x[:, -1, :]  # torch.Size([64, 512])
        output_encoder = x.reshape(x.shape[0],-1)

        # 经过全连接层进行分类任务
        output_encoder = self.fc_task(output_encoder)  # torch.Size([64, 256])
        output = self.classifier(output_encoder)  # torch.Size([64, 2])

        return output , output_encoder.view(output_encoder.size(0),-1)


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size):
        super(MambaBlock, self).__init__()
        self.inp_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(2 * d_model, d_model)
        # For residual skip connection
        self.D = nn.Linear(d_model, 2 * d_model)
        # Set _no_weight_decay attribute on bias
        self.out_proj.bias._no_weight_decay = True
        # Initialize bias to a small constant value
        nn.init.constant_(self.out_proj.bias, 1.0)
        self.S6 = S6(seq_len, 2 * d_model, state_size)
        # Add 1D convolution with kernel size 3
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1)
        # Add linear layer for conv output
        self.conv_linear = nn.Linear(2 * d_model, 2 * d_model)
        # rmsnorm
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
       x_proj.shape = torch.Size([batch_size, seq_len, 2*d_model])
       x_conv.shape = torch.Size([batch_size, seq_len, 2*d_model])
       x_conv_act.shape = torch.Size([batch_size, seq_len, 2*d_model])
       """
        # Refer to Figure 3 in the MAMBA paper
        # x = self.norm(x)
        x_proj = self.inp_proj(x)

        # Add 1D convolution with kernel size 3
        x_conv = self.conv(x_proj)
        x_conv_act = F.silu(x_conv)
        # Add linear layer for conv output
        x_conv_out = self.conv_linear(x_conv_act)
        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)
        # residual skip connection with nonlinearity introduced by multiplication
        x_residual = F.silu(self.D(x))
        x_combined = x_act * x_residual
        x_out = self.out_proj(x_combined)

        return x_out