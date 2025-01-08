import numpy as np
import torch
import time
import torch.nn as nn
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from termcolor import colored
from tqdm import tqdm
from models import Transformer_GRU, GRU, TextCNN, lstm, Another_Transformer
from ernie.dataloader import ac4c_loader
import random
import os

half_length = 500
dataset = 4
batch_size = 4
print("batchsize:", batch_size)
print("dataset:", dataset)
print("left cuda :", torch.cuda.device_count())

device = torch.device("cuda:0")


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):

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


def evaluate(data_iter, model, criterion):
    pred_prob = []
    label_pred = []
    label_real = []

    for j, (data, labels) in enumerate(tqdm(data_iter), 0):
        data = data.to(device)
        labels = labels.to(device)
        output, _ = model(data)
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
train_iter, val_iter, test_iter, max_len = ac4c_loader.load_ac4c_data(halfLength=half_length, batchsize=batch_size,
                                                                      dataset=dataset)

if __name__ == '__main__':
    seed = 42

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model_names = ['Transformer', 'GRU', 'LSTM', 'TextCNN', 'Transformer+GRU']

    model_name = model_names[0]
    early_stopping = EarlyStopping()

    # model = Transformer_GRU.model(415).to(device)
    # model = TextCNN.CnnModel().to(device)
    # model = GRU.model(max_len=max_len).to(device)
    # model = lstm.LSTMModel(max_len=max_len).to(device)
    model = Another_Transformer.Transformer(max_len=max_len).to(device)
    # model = Transformer_GRU.model(max_len=max_len).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-6)

    best_valid_acc = 0
    best_test_acc = 0
    EPOCH = 50

    for epoch in range(EPOCH):
        loss_list = []
        t0 = time.time()
        model.train()

        for i, (data, labels) in enumerate(tqdm(train_iter)):
            data = data.to(device)
            label = labels.to(device)

            logits, _ = model(data)  # torch.Size([b, l])
            optimizer.zero_grad()
            ce_loss = criterion_CE(logits.view(-1, 2), label).mean()

            train_loss = ce_loss

            train_loss.backward(retain_graph=True)

            for name, param in model.named_parameters():
                if 'out_proj.bias' not in name:
                    # clip weights but not bias for out_proj
                    torch.nn.utils.clip_grad_norm_(param, max_norm=10.0)

            optimizer.step()
            loss_list.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data, _ = evaluate(train_iter, model, criterion_CE)
            valid_performance, valid_roc_data, valid_prc_data, valid_loss = evaluate(val_iter, model, criterion_CE)

        # results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_list):.5f}\n"
        results += f'Train: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC, \tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
            valid_performance[4], valid_performance[5]) + '\n' + '=' * 60
        print(results)

        valid_acc = valid_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]

        if valid_acc > best_valid_acc:

            best_valid_acc = valid_acc
            best_epoch = epoch + 1
            best_performance = valid_performance
            filename = '{},{}, {}[{:.4f}].pt'.format(
                'model_{}_dataset={}_length={}'.format(model_name, dataset, half_length * 2 + 1) + ', epoch[{}]'.format(
                    epoch + 1),
                model_name, 'ACC',
                valid_performance[0])
            save_path_pt = os.path.join(f'/home/pod/shared-nvme/Saved_models1113/Models/' + model_name + '/', filename)
            torch.save(model.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)

            save_path_roc = os.path.join(f'/home/pod/shared-nvme/Saved_models1113/ROC/' + model_name + '/', filename)
            save_path_prc = os.path.join(f'/home/pod/shared-nvme/Saved_models1113/PRC/' + model_name + '/', filename)
            torch.save(valid_roc_data, save_path_roc, _use_new_zipfile_serialization=False)
            torch.save(valid_prc_data, save_path_prc, _use_new_zipfile_serialization=False)

        early_stopping(valid_acc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(best_test_acc)