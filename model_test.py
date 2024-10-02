import numpy as np
import torch
import time
import torch.nn as nn
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from termcolor import colored
from tqdm import tqdm
from models import  Transformer_GRU,GRU,TextCNN,lstm,Another_Transformer
from ernie.dataloader import ac4c_loader
import random
import os

half_length = 207
batch_size = 32
print("left cuda :",torch.cuda.device_count())

device = torch.device("cuda:3")

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0


def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    if (tp + fp) == 0:
        PRE = 0
    else:
        PRE = float(tp) / (tp + fp)

    BACC = 0.5 * Sensitivity + 0.5 * Specificity

    performance = [ACC, BACC, Sensitivity, Specificity, MCC, AUC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data

def evaluate(data_iter, net, criterion):
    pred_prob = []
    label_pred = []
    label_real = []

    for j, (data, labels) in enumerate(tqdm(data_iter), 0):
        data = data.to(device)
        labels = labels.to(device)
        output,_ = net(data)
        loss = criterion(output, labels)

        outputs_cpu = output.cpu()
        y_cpu = labels.cpu()
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + output.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()

    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data, loss


def to_log(log):
    with open("./results/ExamPle_Log.log", "a+") as f:
        f.write(log + '\n')


criterion_CE = nn.CrossEntropyLoss()
train_iter , val_iter , test_iter , max_len=  ac4c_loader.load_ac4c_data(halfLength=half_length,batchsize=batch_size)


if __name__ == '__main__':
    seed = 42

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    model_names =['Transformer','GRU','LSTM','TextCNN','Transformer+GRU']

    model_name = model_names[1]
    early_stopping = EarlyStopping()

    # net = Transformer_GRU.model(415).to(device)
    # net = TextCNN.CnnModel().to(device)
    net = GRU.model(max_len=max_len).to(device)
    # net = lstm.LSTMModel(max_len=max_len).to(device)
    # net = Another_Transformer.Transformer(max_len=max_len).to(device)
    # net = Transformer_GRU.model(max_len=max_len).to(device)

    model_path = "/mnt/sdb/home/lrl/code/Saved_models/Models/GRU/model_415, epoch[2],GRU, ACC[0.8595].pt"
    net.load_state_dict(torch.load(model_path))

    net.eval()
    with torch.no_grad():
        test_performance, test_roc_data, test_prc_data, _ = evaluate(test_iter, net, criterion_CE)

    test_results = '\n' + '=' * 16 + colored(' Test Performance. Epoch[{}] ', 'red').format(
        1) + '=' * 16 \
                   + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
        test_performance[0], test_performance[1], test_performance[2], test_performance[3],
        test_performance[4], test_performance[5]) + '\n' + '=' * 60
    print(test_results)