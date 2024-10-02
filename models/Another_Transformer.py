import numpy as np
import torch.nn as nn
import torch

n_layers = 1
n_head = 8
d_model = 256
d_ff = 32
d_k = 32
d_v = 32
device = torch.device("cuda:3")

def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand


class Embedding(nn.Module):
    def __init__(self,max_len):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(6, d_model)  # token embedding (look-up table)
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]

        pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]

        # expand_as类似于expand，只是目标规格是x.shape
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]

        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)

        # layerNorm
        embedding = self.norm(embedding)
        return embedding

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 多头注意力模块
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        # 全连接模块
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs



class Transformer(nn.Module):
    def __init__(self,max_len):
        super(Transformer, self).__init__()

        # transformer encoder
        ways = 2
        self.embedding = Embedding(max_len)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # 定义重复的模块

        # 分类
        self.fc_task = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
        )
        self.classifier = nn.Linear(d_model // 2, ways)

    def forward(self, input_ids, label_input=False, label=None, epoch_num=None):
        # 输入x的形状应该是 [batch_size, sequence_length]
        output_encoder = self.embedding(input_ids)  # torch.Size([64, 301, 512])
        enc_self_attn_mask = get_attn_pad_mask(input_ids)   # torch.Size([64, 301, 301])

        for layer in self.layers:
            output_encoder = layer(output_encoder, enc_self_attn_mask)  # torch.Size([64, 301, 512])

        output_encoder_cls = output_encoder[:, 0, :]    # torch.Size([64, 512])

        # 经过全连接层进行分类任务
        output_encoder = self.fc_task(output_encoder_cls)   # torch.Size([64, 256])
        output = self.classifier(output_encoder)    # torch.Size([64, 2])

        return output,output_encoder_cls.view(output_encoder.size(0),-1)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        # 多头注意力是同时计算的，一次tensor乘法即可，这里是将多头注意力进行切分
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        # context: [batch_size, n_head, seq_len, d_v], attn: [batch_size, n_head, seq_len, seq_len]
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]

        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))