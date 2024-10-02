import pandas as pd
import torch
from models import MydataSet
from torch.utils.data import DataLoader


def process_data(sequences,length):
    new_sequences = []
    for sequence in sequences:
        center_point = len(sequence) // 2  # 计算序列的中心点位置
        new_seq = sequence[center_point - length:center_point + length + 1]  # 取中心点左侧40个字符
        new_sequences.append(new_seq)
    return new_sequences


def load_ac4c_data(index):
    label_enc = {v:k for k,v in enumerate('NATCG')} # Z:0

    train_file0 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/train/{2*index}.csv"
    train_file1 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/train/{2*index+1}.csv"
    val_file0 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/val/{2*index}.csv"
    val_file1 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/val/{2*index+1}.csv"
    test_file0 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/test/{2*index}.csv"
    test_file1 = f"/mnt/sdb/home/lrl/code/Dataset/qjb/test/{2*index+1}.csv"

    train_data0 = pd.read_csv(train_file0)
    train_data1 = pd.read_csv(train_file1)
    val_data0 = pd.read_csv(val_file0)
    val_data1 = pd.read_csv(val_file1)
    test_data0 = pd.read_csv(test_file0)
    test_data1 = pd.read_csv(test_file1)

    train_seq  = list(train_data0['data']) + (list(train_data1['data']))
    train_label = list(train_data0['label']) + (list(train_data1['label']))
    val_seq = list(val_data0['data'])+(list(val_data1['data']))
    val_label =  list(val_data0['label']) + list(val_data1['label'])
    test_seq  = list(test_data0['data'])+(list(test_data1['data']))
    test_label = list(test_data0['label'])  +  (list(test_data1['label']))

    # 0 - 206
    # max_len = 1001
    half_length = 500

    train_seq = process_data(train_seq,half_length)
    test_seq = process_data(test_seq,half_length)
    val_seq = process_data(val_seq,half_length)

    train_seq = [torch.tensor([label_enc[i] for i in seq])for seq in train_seq]
    val_seq = [torch.tensor([label_enc[i] for i in seq])for seq in val_seq]
    test_seq = [torch.tensor([label_enc[i] for i in seq])for seq in test_seq]

    train_dataset = MydataSet.MyDataSet(train_seq,train_label)
    val_dataset = MydataSet.MyDataSet(val_seq, val_label)
    test_dataset = MydataSet.MyDataSet(test_seq, test_label)

    train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size=16,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=True,drop_last=True)

    return train_dataloader , val_dataloader ,test_dataloader ,2 * half_length +1

