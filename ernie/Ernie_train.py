import sys
import random
from termcolor import colored
import torch.utils.data as Data
from dataloader import ernie_loader
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from src.ernie_rna.models.ernie_rna import *
from tqdm import tqdm
from src.ernie_rna.criterions.ernie_rna import *
from src.utils import ErnieRNAOnestage, load_pretrained_ernierna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsize = 4
half_length = 50
dataset = 10

print("batchsize:", batchsize)
print("dataset:", dataset)
print("start")
print("build model")
print("left cuda:", torch.cuda.device_count())
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "cola"
pretrained_model_path = "/home/pod/shared-nvme/ERNIE-RNA/checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt"
arg_overrides = {'data': '/home/pod/shared-nvme/ERNIE-RNA/src/dict/'}


def to_log(log):
    with open(f"./results/Ernie_result_dataset=" + str(dataset) + "_length=" + str(half_length * 2 + 1) + ".log",
              "a+") as f:
        f.write(log + '\n')


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

    for j, (one_d, two_d, labels) in enumerate(tqdm(data_iter), 0):
        one_d = one_d.to(device)
        two_d = two_d.to(device)
        labels = labels.to(device)

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


def load_pretrained_ernierna(mlm_pretrained_model_path, arg_overrides):
    rna_models, _, _ = checkpoint_utils.load_model_ensemble_and_task(mlm_pretrained_model_path.split(os.pathsep),
                                                                     arg_overrides=arg_overrides)
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


class Ernierna(nn.Module):
    def __init__(self, device='cuda:3'):
        super(Ernierna, self).__init__()
        self.hidden_dim = 25
        self.emb_dim = 768

        self.device = device

        self.model_pretrained = load_pretrained_ernierna(pretrained_model_path, arg_overrides)
        self.ernie = ErnieRNAOnestage(self.model_pretrained.encoder).to(device)

        # self.GRU = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2, bidirectional=True,dropout=0.2)

        # self.block = nn.Sequential(
        #    nn.Linear(self.emb_dim * (half_length * 2 + 3), 1024),  #415:318720 20850 201:154368 10250 101:79104 51:40704 39168 2650
        #    nn.BatchNorm1d(1024),
        #    nn.LeakyReLU(),
        #    nn.Linear(1024, 128),
        #    nn.BatchNorm1d(128),
        #    nn.LeakyReLU(),
        #    nn.Linear(128, 2),
        # )
        self.block = nn.Sequential(
            nn.Linear(768, 64),  # 415:318720 20850 201:154368 10250 101:79104 51:40704 39168 2650
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, one_d, two_d):  # torch.Size([16, 1, length + 2])
        embedding = []

        for od, td in zip(one_d, two_d):
            output = self.ernie(od, td)
            embedding.append(output)

        embedding = torch.stack(embedding, dim=0).to(device)  # 先将所有张量堆叠，然后移动到 GPU 上
        embedding = embedding.squeeze(dim=1)  # torch.Size([8, length + 2, 768])
        # print(embedding.shape)

        hidden_state = embedding[:, 0, :]

        # hidden_state = embedding.view(embedding.shape[0],-1)

        # 分类器
        output = self.block(hidden_state)

        return output


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

    # 参数更改：ernie_loader的长度，erniedataset的length，模型的分类层的神经元个数，保存的文件名和文件路径
    SEED = 42
    set_seed(seed=SEED)

    train_loader, valid_loader, test_loader = ernie_loader.ernie_load_ac4c_data(batchsize=batchsize,
                                                                                halfLength=half_length, dataset=dataset)

    epoch_num = 50

    print('training...')

    # load model
    model = Ernierna(device=device).to(device)
    print('Model Loading Done!!!')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping()

    best_acc = 0
    best_test_acc = 0
    best_valid_acc = 0
    best_epoch = 0
    print("模型数据已经加载完成,现在开始模型训练。")

    total_outputs = []  # logits
    total_labels = []  # labels

    for epoch in range(epoch_num):
        total_outputs = []  # logits
        total_labels = []  # labels
        loss_ls = []
        t0 = time.time()
        model.train()
        for i, (one_d, two_d, labels) in enumerate(tqdm(train_loader)):
            one_d = one_d.to(device)
            two_d = two_d.to(device)
            labels = labels.to(device)

            output = model(one_d, two_d)
            optimizer.zero_grad()  # 梯度清0
            loss = criterion(output, labels)  # 计算误差
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            loss_ls.append(loss.item())

            # 将输出和标签添加到总列表中
            total_outputs.append(output.detach().cpu().numpy())
            total_labels.append(labels.detach().cpu().numpy())

        # 在每个 epoch 结束后计算总体准确率
        total_outputs = np.concatenate(total_outputs, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        predicted_labels = np.argmax(total_outputs, axis=1)
        train_acc = np.mean(predicted_labels == total_labels)

        print('testing...')

        model.eval()
        with torch.no_grad():
            valid_performance, valid_roc_data, valid_prc_data, valid_loss = evaluate(valid_loader, model, criterion)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'Train: {train_acc:.4f}, time: {time.time() - t0:.2f}'
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
            filename = '{},{},[{:.4f}].pt'.format(
                'Ernie_model_{}_SEED={}_dataset={}'.format(half_length * 2 + 1, SEED, dataset) + ', epoch[{}]'.format(
                    epoch + 1), 'ACC',
                valid_performance[0])
            save_path_pt = os.path.join(f'/home/pod/shared-nvme/Saved_models1113/Models/Ernie/', filename)
            torch.save(model.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)

            save_path_roc = os.path.join(f'/home/pod/shared-nvme/Saved_models1113/ROC/Ernie/', filename)
            save_path_prc = os.path.join(f'/home/pod/shared-nvme/Saved_models1113/PRC/Ernie/', filename)
            torch.save(valid_roc_data, save_path_roc, _use_new_zipfile_serialization=False)
            torch.save(valid_prc_data, save_path_prc, _use_new_zipfile_serialization=False)

        early_stopping(valid_acc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print(best_test_acc)



