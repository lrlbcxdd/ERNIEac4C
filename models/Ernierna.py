import torch.nn as nn
from termcolor import colored
import torch.utils.data as Data
from dataloader import ernie_loader
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

from finetuning.ernie.src.ernie_rna.tasks.ernie_rna import *
from finetuning.ernie.src.ernie_rna.models.ernie_rna import *
from finetuning.ernie.src.ernie_rna.criterions.ernie_rna import *
from finetuning.ernie.src.utils import ErnieRNAOnestage, read_text_file, load_pretrained_ernierna, prepare_input_for_ernierna

pretrained_model_path = "/mnt/sdb/home/lrl/code/git_ERNIE/ernie/checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt"
arg_overrides = {'data': '/mnt/sdb/home/lrl/code/git_ERNIE/ernie/src/dict/'}

def load_pretrained_ernierna(mlm_pretrained_model_path,arg_overrides):
    rna_models, _, _ = checkpoint_utils.load_model_ensemble_and_task(mlm_pretrained_model_path.split(os.pathsep),arg_overrides=arg_overrides)
    model_pretrained = rna_models[0]
    return model_pretrained

class Ernierna(nn.Module):
    def __init__(self, hidden_size=128, device='cuda'):

        super(Ernierna, self).__init__()
        self.hidden_dim = 25
        self.emb_dim = 256

        self.device = device

        self.model_pretrained = load_pretrained_ernierna(pretrained_model_path, arg_overrides)
        self.ernie = ErnieRNAOnestage(self.model_pretrained.encoder).to(device)

        self.GRU = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2, bidirectional=True,dropout=0.2)

        # self.block = nn.Sequential(
        #     nn.Linear(256, 64),  # 1001:128384  51:6784  510：65536
        #     nn.Dropout(0.2),
        #     nn.LeakyReLU(),
        #     nn.Linear(64,2)
        # )

        self.block = nn.Sequential(
            nn.Linear(100250, 1024),  # 1001:128384  51:6784  510：65536 GRU:50150  lsr:128512
            nn.BatchNorm1d(1024),
            # nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )


    def forward(self, one_d,two_d):  # [batch_size,1]

        output_embedding = self.ernie(one_d,two_d)

        # 不使用GRU
        # hidden_state = hidden_state[:,1,:]
        hidden_state = output_embedding.view(output_embedding.shape[0],-1)

        # 用GRU
        # x = hidden_state.permute(1, 0, 2)  # torch.Size([128, 1001, 128])   torch.Size([32, 2003, 256])
        # output, hn = self.GRU(x)  # output: torch.Size([max_len, 128, hidden_dim*2]) hn: torch.Size([4, 128, 25])
        # output = output.permute(1, 0, 2)  # output: torch.Size([128, max_len, hidden_dim*2])
        # hn = hn.permute(1, 0, 2)  # torch.Size([128, 4, hidden_dim])
        #
        # output = output.reshape(output.shape[0], -1)  # torch.Size([128, max_len * hidden_dim *2])
        # hn = hn.reshape(output.shape[0], -1)  # torch.Size([128, 4* hidden_dim])
        # hidden_state = torch.cat([output, hn], 1)  # output:torch.Size([128, hidden_dim *(2*max_len + 4)])


        # 分类器
        output = self.block(hidden_state)   # torch.Size([32, 50150])   torch.Size([32, 100250])


        return output
