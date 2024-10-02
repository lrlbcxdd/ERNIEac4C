import pandas as pd
import ClsDataset
from torch.utils.data import DataLoader
import numpy as np


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

    train_dataset = ClsDataset.MyDataset(train_seq,train_label)
    val_dataset = ClsDataset.MyDataset(val_seq, val_label)
    test_dataset = ClsDataset.MyDataset(test_seq, test_label)

    train_dataloader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size=batchsize,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset,batch_size=batchsize,shuffle=True,drop_last=False)

    return train_dataloader , val_dataloader ,test_dataloader


