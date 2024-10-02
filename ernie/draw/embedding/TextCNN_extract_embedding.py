
import random

import torch
from termcolor import colored
import torch.utils.data as Data
from ernie.dataloader import ac4c_loader

from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from ernie.src.ernie_rna.models.ernie_rna import *
from tqdm import tqdm
from ernie.src.ernie_rna.criterions.ernie_rna import *
from ernie.src.utils import ErnieRNAOnestage, load_pretrained_ernierna
from models import TextCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
dataset = 4
batchsize = 4
half_length = 207
print("batchsize:",batchsize)
print("start")
print("build model")
print("left cuda:",torch.cuda.device_count())
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "cola"
pretrained_model_path = "/mnt/sdb/home/lrl/code/git_ERNIE/ERNIE-RNA/checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt"
arg_overrides = {'data': '/mnt/sdb/home/lrl/code/git_ERNIE/ERNIE-RNA/src/dict/'}

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



class RNADataset(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = torch.tensor(label)

    def __getitem__(self, i):
        return self.data[i], self.label[i]

    def __len__(self):
        return len(self.label)


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

    for j, (one_d, two_d ,labels) in enumerate(tqdm(data_iter), 0):
        one_d = one_d.to('cuda:3')
        two_d = two_d.to('cuda:3')
        labels = labels.to('cuda:3')

        output = net(one_d, two_d)
        loss = criterion(output, labels)

        outputs_cpu = output.cpu()
        y_cpu = labels.cpu()
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + output.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data, loss

def load_pretrained_ernierna(mlm_pretrained_model_path,arg_overrides):
    rna_models, _, _ = checkpoint_utils.load_model_ensemble_and_task(mlm_pretrained_model_path.split(os.pathsep),arg_overrides=arg_overrides)
    model_pretrained = rna_models[0]
    return model_pretrained

# Note that load_metric has loaded the proper metric associated to your task, which is:
task_to_keys = {
    "cola": ("data", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

num_labels = 2
metric_name = "accuracy"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

if __name__ == '__main__':

    SEED = 29
    set_seed(seed= SEED)

    criterion_CE = nn.CrossEntropyLoss()
    train_iter, val_iter, test_iter, max_len = ac4c_loader.load_ac4c_data(halfLength=207, batchsize=1)

    print('extracting...')

    # load model
    model = TextCNN.CnnModel().to('cuda:3')
    print('Model Loading Done!!!')

    model_path = "/mnt/sdb/home/lrl/code/Saved_models/Models/LSTM/model_415, epoch[7],LSTM, ACC[0.8622].pt"
    model.load_state_dict(torch.load(model_path))

    print("模型数据已经加载完成,现在开始模型训练。")

    all_embedding = torch.Tensor().to('cuda:3')
    all_labels = torch.Tensor().to('cuda:3')

    print('testing...')

    model.eval()
    with torch.no_grad():
        for j, (data, labels) in enumerate(tqdm(test_iter)):
            data = data.to('cuda:3')
            labels = labels.to('cuda:3')
            logits , output = model(data)
            # output = nn.BatchNorm1d(output)
            all_embedding = torch.cat((all_embedding, output), dim=0)
            all_labels = torch.cat((all_labels,labels),dim=0)

    torch.save(all_embedding, '../pt/TextCNN_embedding.pt')
    torch.save(all_labels, '../pt/TextCNN_labels.pt')
