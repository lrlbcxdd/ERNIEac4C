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


length = 415

def prepare_input_for_ernierna(index):
    print("zheshilength" , length)
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

