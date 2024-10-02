
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.utils.data as Data


from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from ernie.src.ernie_rna.models.ernie_rna import *
from tqdm import tqdm
from ernie.src.ernie_rna.criterions.ernie_rna import *
from ernie.src.utils import ErnieRNAOnestage, load_pretrained_ernierna


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
    rna_index = np.ones((seq_nums, max_len))
    for i in range(seq_nums):
        for j in range(rna_len_lst[i]):
            if sequences[i][j] in set("Aa"):
                rna_index[i][j] = 5
            elif sequences[i][j] in set("Cc"):
                rna_index[i][j] = 7
            elif sequences[i][j] in set("Gg"):
                rna_index[i][j] = 4
            elif sequences[i][j] in set('TUtu'):
                rna_index[i][j] = 6
            else:
                rna_index[i][j] = 3
    #     rna_index[i][rna_len_lst[i] + 1] = 2  # add 'eos' token
    # rna_index[:, 0] = 0  # add 'cls' token
    return rna_index, max_len

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

    train_dataloader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size=batchsize,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset,batch_size=batchsize,shuffle=True,drop_last=False)

    return train_dataloader , val_dataloader ,test_dataloader



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

class Ernierna(nn.Module):
    def __init__(self,):
        super(Ernierna, self).__init__()
        self.length = 415

        self.hidden_dim = 25
        self.emb_dim = 768
        self.attn_len = 51


        self.model_pretrained = load_pretrained_ernierna(pretrained_model_path, arg_overrides)
        self.ernie = ErnieRNAOnestage(self.model_pretrained.encoder).to('cuda:3')

        self.filter_sizes = [2, 4, 8, 16, 32, 64, 128, 256]   # 0.9297


        self.num_filters = 64

        self.convs_emb = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.emb_dim)) for k in self.filter_sizes])
        self.convs_attn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.length)) for k in self.filter_sizes])

        self.fc = nn.Sequential(
            nn.Linear(len(self.filter_sizes) * self.num_filters * 2, len(self.filter_sizes) * self.num_filters),
            nn.BatchNorm1d(len(self.filter_sizes) * self.num_filters),
            nn.ReLU(),
            nn.Linear(len(self.filter_sizes) * self.num_filters, 2)
        )


    def forward(self, one_d, two_d):  # torch.Size([16, 1, length + 2])
        embedding = torch.Tensor().to('cuda:3')
        rna_attn_map_embedding = torch.Tensor().to('cuda:3')

        for od, td in zip(one_d, two_d):
            output = self.ernie(od, td)  # torch.Size([1, length, dmodel])
            embedding = torch.cat((embedding, output), dim=0)
            attn_output = self.ernie(od, td, return_attn_map=True)  # torch.Size([length, length])
            rna_attn_map_embedding = torch.cat((rna_attn_map_embedding, attn_output.unsqueeze(dim=0)), dim=0)


        embedding = embedding.unsqueeze(dim=1)  # torch.Size([batch size,1, length, dmodel])
        embedding = [torch.tanh(conv(embedding)).squeeze(3) for conv in self.convs_emb]
        embedding = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in embedding]
        embedding = torch.cat(embedding, 1) # torch.Size([batch size, 512])

        rna_attn_map_embedding = rna_attn_map_embedding.unsqueeze(dim=1)    # torch.Size([batchsize, 1 ,length, length])
        rna_attn_map_embedding = [torch.tanh(conv(rna_attn_map_embedding)).squeeze(3) for conv in self.convs_attn]
        rna_attn_map_embedding = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in rna_attn_map_embedding]
        rna_attn_map_embedding = torch.cat(rna_attn_map_embedding, 1)   # torch.Size([batch size, 512])

        # 分类器
        output = self.fc[:2](torch.cat((embedding, rna_attn_map_embedding), dim=1))

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

    SEED = 29
    set_seed(seed= SEED)

    train_loader , valid_loader , test_loader =  ernie_load_ac4c_data(batchsize=batchsize,halfLength=half_length)

    print('extracting...')

    # load model
    model = Ernierna().to('cuda:3')
    print('Model Loading Done!!!')

    model_path = "/mnt/sdb/home/lrl/code/Saved_models/Models/Attn/Test_Atten_model_415_SEED=29_data4, epoch[23],RNA, ACC[0.9324].pt"
    model.load_state_dict(torch.load(model_path))
    print("模型数据已经加载完成,现在开始模型训练。")

    all_embedding = torch.Tensor().to('cuda:3')
    all_labels = torch.Tensor().to('cuda:3')

    print('testing...')

    model.eval()
    with torch.no_grad():
        for j, (one_d, two_d, labels) in enumerate(tqdm(test_loader), 0):
            one_d = one_d.to('cuda:3')
            two_d = two_d.to('cuda:3')
            labels = labels.to('cuda:3')
            output = model(one_d, two_d)
            all_embedding = torch.cat((all_embedding, output), dim=0)
            all_labels = torch.cat((all_labels,labels),dim=0)

    torch.save(all_embedding, '../pt/all_embeddings_2.pt')
    torch.save(all_labels, '../pt/all_labels_2.pt')
