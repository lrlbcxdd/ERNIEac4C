import torch.nn as nn
import torch
import torch.nn.functional as F

class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        ways=2

        dim_cnn_out = 128
        filter_num = 32
        filter_sizes = [1,2,4,8,16,24]
        vocab_size = 5
        embedding_dim = 128

        self.embedding_cnn = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, embedding_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(0.4)

        self.linear_cnn = nn.Linear(len(filter_sizes) * filter_num, dim_cnn_out)

        # 分类
        self.fc_task = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),  # 调整输入维度
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),  # 调整输出维度
        )
        self.classifier = nn.Linear(embedding_dim // 4, ways)  # 调整输出维度

    def forward(self, input_ids, label_input=False, label=None, epoch_num=None):
        feature_cnn = self.embedding_cnn(input_ids)
        feature_cnn = feature_cnn.view(feature_cnn.size(0), 1, feature_cnn.size(1),self.embedding_cnn.embedding_dim)
        feature_cnn = [F.relu(conv(feature_cnn)) for conv in self.convs]
        feature_cnn = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in
                       feature_cnn]
        feature_cnn = [x_item.view(x_item.size(0), -1) for x_item in feature_cnn]
        feature_cnn = torch.cat(feature_cnn, 1)
        feature_cnn = self.linear_cnn(feature_cnn)

        feature_new = self.fc_task(feature_cnn)
        logits_clsf = self.classifier(feature_new)
        embeddings = feature_new.view(feature_new.size(0), -1)
        return logits_clsf, feature_cnn
