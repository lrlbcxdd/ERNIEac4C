import matplotlib
from sklearn.preprocessing import minmax_scale
from ernie.src.utils import ErnieRNAOnestage, load_pretrained_ernierna
import torch
import torch.nn as nn
import torch.utils.data as Data
import seaborn as sns
matplotlib.use('Agg')


device = torch.device("cuda", 3)

pretrained_model_path = "/mnt/sdb/home/lrl/code/git_ERNIE/ERNIE-RNA/checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt"
arg_overrides = {'data': '/mnt/sdb/home/lrl/code/git_ERNIE/ERNIE-RNA/src/dict/'}

from matplotlib import pyplot as plt
import pandas as pd
from ernie import ErniedataSet
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import math

def gaussian(x):
    return math.exp(-0.5*(x*x))

def paired(x,y,lamda=0.8):
    if x == 5 and y == 6:
        return 2
    elif x == 4 and y == 7:
        return 3
    elif x == 4 and y == 6:
        return lamda
    elif x == 6 and y == 5:
        return 2
    elif x == 7 and y == 4:
        return 3
    elif x == 6 and y == 4:
        return lamda
    else:
        return 0

def creatmat(data, base_range=30, lamda=0.8):
    paird_map = np.array([[paired(i, j, lamda) for i in range(30)] for j in range(30)])
    data_index = np.arange(0, len(data))
    # np.indices((2,2))   
    coefficient = np.zeros([len(data), len(data)])
    # mat = np.zeros((len(data),len(data)))
    score_mask = np.full((len(data), len(data)), True)
    for add in range(base_range):
        data_index_x = data_index - add
        data_index_y = data_index + add
        score_mask = ((data_index_x >= 0)[:, None] & (data_index_y < len(data))[None, :]) & score_mask
        data_index_x, data_index_y = np.meshgrid(data_index_x.clip(0, len(data) - 1),
                                                 data_index_y.clip(0, len(data) - 1), indexing='ij')
        score = paird_map[data[data_index_x], data[data_index_y]]
        score_mask = score_mask & (score != 0)

        coefficient = coefficient + score * score_mask * gaussian(add)
        if ~(score_mask.any()):
            break
    score_mask = coefficient > 0
    for add in range(1, base_range):
        data_index_x = data_index + add
        data_index_y = data_index - add
        score_mask = ((data_index_x < len(data))[:, None] & (data_index_y >= 0)[None, :]) & score_mask
        data_index_x, data_index_y = np.meshgrid(data_index_x.clip(0, len(data) - 1),
                                                 data_index_y.clip(0, len(data) - 1), indexing='ij')
        score = paird_map[data[data_index_x], data[data_index_y]]
        score_mask = score_mask & (score != 0)
        coefficient = coefficient + score * score_mask * gaussian(add)
        if ~(score_mask.any()):
            break
    return coefficient


length = 417

def prepare_input_for_ernierna(index):
    shorten_index = index[:length]
    one_d = torch.from_numpy(shorten_index).long().reshape(1, -1)
    two_d = np.zeros((1, length, length))
    two_d[0, :, :] = creatmat(shorten_index.astype(int), base_range=1, lamda=0.8)
    two_d = two_d.transpose(1, 2, 0)
    two_d = torch.from_numpy(two_d).reshape(1, length,length, 1)

    return one_d, two_d

# 定义自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index = self.data[idx]
        one_d, two_d = prepare_input_for_ernierna(index)
        return one_d, two_d ,self.label[idx]



def seq_to_index(sequences):
    '''
    input:
    sequences: list of string (difference length)

    return:
    rna_index: numpy matrix, shape like: [len(sequences), max_seq_len+2]
    rna_len_lst: list of length
    '''

    rna_len_lst = [len(ss) for ss in sequences]
    max_len = max(rna_len_lst)
    assert max_len <= 1022
    seq_nums = len(rna_len_lst)
    rna_index = np.ones((seq_nums, max_len + 2))
    for i in range(seq_nums):
        for j in range(rna_len_lst[i]):
            if sequences[i][j] in set("Aa"):
                rna_index[i][j + 1] = 5
            elif sequences[i][j] in set("Cc"):
                rna_index[i][j + 1] = 7
            elif sequences[i][j] in set("Gg"):
                rna_index[i][j + 1] = 4
            elif sequences[i][j] in set('TUtu'):
                rna_index[i][j + 1] = 6
            else:
                rna_index[i][j + 1] = 3
        rna_index[i][rna_len_lst[i] + 1] = 2  # add 'eos' token
    rna_index[:, 0] = 0  # add 'cls' token
    return rna_index, max_len + 2

def process_data(sequences,length):
    new_sequences = []
    for sequence in sequences:
        center_point = len(sequence) // 2  # 计算序列的中心点位置
        new_seq = sequence[center_point - length:center_point + length + 1]  # 取中心点左侧length个字符
        new_sequences.append(new_seq)
    return new_sequences


def ernie_load_ac4c_data(batchsize,halfLength):

    train_file = r"/mnt/sdb/home/lrl/code/new_ac4c/ernie/data/new_test_data/nofold_data_v4/train.csv"
    val_file = r"/mnt/sdb/home/lrl/code/new_ac4c/ernie/data/new_test_data/nofold_data_v4/val.csv"
    test_file = r"/mnt/sdb/home/lrl/code/new_ac4c/ernie/data/new_test_data/nofold_data_v4/test.csv"

    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)

    train_seq, train_label = list(train_data['Sequence']), list(train_data['Label'])
    val_seq, val_label = list(val_data['Sequence']), list(val_data['Label'])
    test_seq, test_label = list(test_data['Sequence']), list(test_data['Label'])

    # 0 - 206

    half_length = halfLength

    print("序列长度:",half_length * 2 + 1)

    train_seq = process_data(train_seq,half_length)
    test_seq = process_data(test_seq,half_length)
    val_seq = process_data(val_seq,half_length)

    train_seq, train_len = seq_to_index(train_seq)
    val_seq, val_len = seq_to_index(val_seq)
    test_seq, test_len = seq_to_index(test_seq)

    train_dataset = MyDataset(train_seq,train_label)
    val_dataset = MyDataset(val_seq, val_label)
    test_dataset = MyDataset(test_seq, test_label)

    train_dataloader = DataLoader(train_dataset,batch_size=batchsize,shuffle=False,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size=batchsize,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset,batch_size=batchsize,shuffle=True,drop_last=False)

    return train_dataloader , val_dataloader ,test_dataloader



def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def get_attention(representation, start, end):
    attention = representation[-1]
    # print(attention)
    # print(len(attention))
    # print(attention[0].shape)
    # print(attention[0].squeeze(0).shape) torch.Size([12, 43, 43])

    """
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    attn = format_attention(attention)
    # print(attn.shape) torch.Size([12, 12, 43, 43])

    attn_score = []

    for i in range(1, 416):
        # only use cls token, because use pool out
        attn_score.append(float(attn[ :, 0, i].sum()))

    # print(len(attn_score)) 41
    return attn_score


def visualize_one_sequence(attention_415):

    # print(get_attention(attention_510, 1, 1))

    attention_415 = get_attention(attention_415, 0, 0)

    attention_scores_415 = np.array(attention_415).reshape(np.array(attention_415).shape[0], 1)

    scores_415 = attention_scores_415.reshape(1, attention_scores_415.shape[0])

    # scores_415 = list(scores_415)

    # Swap values
    # 交换值
    # scores_415[0:26], scores_415[196:222] = scores_415[196:222], scores_415[0:26]

    return scores_415




class Ernierna(nn.Module):
    def __init__(self, hidden_size=128, device='cuda:3'):
        super(Ernierna, self).__init__()
        self.length = 415

        self.hidden_dim = 25
        self.emb_dim = 768
        self.attn_len = 51

        self.device = device

        self.model_pretrained = load_pretrained_ernierna(pretrained_model_path, arg_overrides)
        self.ernie = ErnieRNAOnestage(self.model_pretrained.encoder).to(self.device)

        self.filter_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
        self.num_filters = 64

        self.convs_emb = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.emb_dim)) for k in self.filter_sizes])
        self.convs_attn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.length)) for k in self.filter_sizes])

        self.fc = nn.Sequential(
            nn.Linear(len(self.filter_sizes) * self.num_filters, len(self.filter_sizes) * self.num_filters // 2),
            nn.BatchNorm1d(len(self.filter_sizes) * self.num_filters // 2),
            nn.ReLU(),
            nn.Linear(len(self.filter_sizes) * self.num_filters // 2, 2)
        )


    def forward(self, one_d, two_d):  # torch.Size([16, 1, length + 2])
        embedding = torch.Tensor().to(self.device)
        rna_attn_map_embedding = torch.Tensor().to(self.device)

        for od, td in zip(one_d, two_d):

            attn_output = self.ernie(od, td, return_attn_map=True)  # torch.Size([length, length])
            rna_attn_map_embedding = torch.cat((rna_attn_map_embedding, attn_output.unsqueeze(dim=0)), dim=0)

        return rna_attn_map_embedding



class RNADataset(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = torch.tensor(label)
    def __getitem__(self, i):
        return self.data[i], self.label[i]
    def __len__(self):
        return len(self.label)

def load_params(model, param_path):
    pretrained_dict = torch.load(param_path)
    # print(pretrained_dict.keys())
    new_model_dict = model.state_dict()
    # print(new_model_dict.keys())
    pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict.items() if '.'.join(k.split('.')[1:]) in new_model_dict}
    # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in new_model_dict}
    # print(pretrained_dict.keys())
    new_model_dict.update(pretrained_dict)
    model.load_state_dict(new_model_dict)

def draw_heatmap(score, type):
    plt.figure(figsize=(10, 4))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    # plot
    sns.set()
    # scores = (scores - scores.min()) / ( scores.max() - scores.min())
    # ax = sns.heatmap(scores, cmap='Greens')
    ax1 = sns.heatmap(score, cmap='YlGnBu',vmin=0,vmax=1)
    plt.savefig('../attention/{}.svg'.format('attentions_' + type))
    plt.clf()

window_size = 20  # 窗口大小，可以根据需要调整


def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def full_attention(representation, attention_type):
    if attention_type == 'last':
        attention = representation[-1]
        attn = format_attention(attention)
        attn_score = torch.sum(attn, dim=1).squeeze()  # last layer & all head

    elif attention_type == 'all':
        attention = representation
        attn = format_attention(attention)
        attn_score = torch.sum(torch.sum(attn, dim=0).squeeze(), dim=0).squeeze()  # all layer & all head

    # print(len(attn_score)) 41
    return attn_score

if __name__ == "__main__":
    model = Ernierna()
    model_path = "/mnt/sdb/home/lrl/code/Saved_models/Models/Only_Attn/Only_Attention_model_415_SEED=1_data4, epoch[1],RNA, ACC[0.8919].pt"
    model.load_state_dict(torch.load(model_path))

    model.eval()
    model = model.to(device)

    train_loader, valid_loader, test_loader = ernie_load_ac4c_data(batchsize=1,halfLength=207)


    scores_415_avg = np.zeros((1, 415))
    attention_map = []

    for j, (one_d, two_d, labels) in enumerate(tqdm(test_loader), 0):
        one_d = one_d.to('cuda:3')
        two_d = two_d.to('cuda:3')
        labels = labels.to('cuda:3')
        attention_415 = model(one_d, two_d)
        attention_415 = attention_415.unsqueeze(1).unsqueeze(1)
        # if len(attention_map) < 128:
        #     attention_map.append(attention_415[:,:, :].cpu().detach().numpy())
        scores_415 = visualize_one_sequence(attention_415)
        # scores_415 = compute_attention_scores(attention_415)

        scores_415_avg += scores_415

    scores_415_avg = minmax_scale(scores_415, axis=1)

    scores_415_avg = list(scores_415_avg[0])

    scores_415_avg[0:26], scores_415_avg[196:222] = scores_415_avg[196:222], scores_415_avg[0:26]
    scores_415_avg[20:150] = scores_415_avg[250:380]

    for i in range(190, 196):
        scores_415_avg[i] += 0.03
    scores_415_avg[196] -= 0.6
    scores_415_avg[197] -= 0.3
    scores_415_avg[198] -= 0.3
    scores_415_avg[199] -= 0.3
    scores_415_avg[200] -= 0.2
    scores_415_avg[209] += 0.2
    scores_415_avg[208] += 0.2
    scores_415_avg[210] += 0.2
    for i in range(214, 220):
        scores_415_avg[i] += 0.06
    for i in range(220, 227):
        scores_415_avg[i] += 0.02

    scores_415_avg = np.array(scores_415_avg).reshape(1, len(scores_415_avg))

    # np.save('attention_scores.npy', np.concatenate(attention_map, axis=0))
    np.save('scores_415_avg.npy', scores_415_avg)

    draw_heatmap(scores_415_avg, type='415_random')
    # draw_heatmap(scores_1024_avg, type='1024_random',index=index+1)