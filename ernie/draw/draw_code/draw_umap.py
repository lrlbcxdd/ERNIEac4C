import torch
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.use('Agg')

color1 = '#01a9ce'
color2 = '#abdfeb'

plt.rcParams['font.family'] = 'Times New Roman'

def ernie_ac4c_draw_umap():

    x = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/all_embeddings_2.pt") #(num,128)的特征
    y = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/all_labels_2.pt")# (num,2)的标签

    cmap_colors = [color1, color2]
    cmap = ListedColormap(cmap_colors)

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    reducer = umap.UMAP(random_state=29)
    embedding = reducer.fit_transform(x)    # umap降维从128维度成2维度

    # 筛选掉x坐标小于0的数据点
    mask = embedding[:, 0] <= 15
    embedding_filtered = embedding[mask]
    y_filtered = y[mask]

    np.save('1embedding_filtered.npy', embedding_filtered)
    np.save('1y_filtered.npy', y_filtered)

    plt.scatter(embedding_filtered[:, 0], embedding_filtered[:, 1], c=y_filtered,cmap=cmap, s=10) #画散点图
    plt.gca().set_aspect('equal', 'datalim')

    c = plt.colorbar(ticks=[0, 1])  # 设置颜色条的刻度位置
    c.ax.tick_params(labelsize=15)
    c.set_ticks([0.25, 0.75])  # 设置刻度为柱子的四分之一和四分之三处
    c.set_ticklabels(['pos', 'neg'])  # 设置刻度标签为 "neg" 和 "pos"

    del c

    plt.title(f'ERNIE-ac4C', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 保存图像时指定格式为SVG
    plt.savefig(f'../umap_picture/ernie_ac4c.svg', format='svg', dpi=300)
    plt.show()

def transformer_draw_umap():

    x = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/Transformer_embedding.pt")
    y = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/Transformer_labels.pt")

    cmap_colors = [color1, color2]
    cmap = ListedColormap(cmap_colors)

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    reducer = umap.UMAP(random_state=29)
    embedding = reducer.fit_transform(x)

    np.save('2embedding_filtered.npy', embedding)
    np.save('2y_filtered.npy', y)

    # mask = embedding[:, 0] >= 0
    # embedding_filtered = embedding[mask]
    # y_filtered = y[mask]

    plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap=cmap, s=10)
    plt.gca().set_aspect('equal', 'datalim')

    c = plt.colorbar(ticks=[0, 1])  # 设置颜色条的刻度位置
    c.ax.tick_params(labelsize=15)
    c.set_ticks([0.25, 0.75])  # 设置刻度为柱子的四分之一和四分之三处
    c.set_ticklabels(['pos', 'neg'])  # 设置刻度标签为 "neg" 和 "pos"
    #
    # del c

    plt.title(f'Transformer', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 保存图像时指定格式为SVG
    plt.savefig(f'../umap_picture/transformer.svg', format='svg', dpi=300)
    plt.show()

def gru_draw_umap():

    x = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/GRU_embedding.pt")
    y = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/GRU_labels.pt")

    cmap_colors = [color1, color2]
    cmap = ListedColormap(cmap_colors)

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    reducer = umap.UMAP(random_state=29)
    embedding = reducer.fit_transform(x)

    # 筛选掉x坐标小于0的数据点

    mask = embedding[:, 0] >= -15
    embedding_filtered = embedding[mask]
    y_filtered = y[mask]

    np.save('3embedding_filtered.npy', embedding_filtered)
    np.save('3y_filtered.npy', y_filtered)

    plt.scatter(embedding_filtered[:, 0], embedding_filtered[:, 1], c=y_filtered, cmap=cmap, s=10)
    plt.gca().set_aspect('equal', 'datalim')

    c = plt.colorbar(ticks=[0, 1])  # 设置颜色条的刻度位置
    c.ax.tick_params(labelsize=15)
    c.set_ticks([0.25, 0.75])  # 设置刻度为柱子的四分之一和四分之三处
    c.set_ticklabels(['pos', 'neg'])  # 设置刻度标签为 "neg" 和 "pos"
    #
    # del c

    plt.title(f'GRU', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 保存图像时指定格式为SVG
    plt.savefig(f'../umap_picture/gru.svg', format='svg', dpi=300)
    plt.show()

def lstm_draw_umap():

    x = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/lstm_embedding.pt")
    y = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/lstm_labels.pt")

    cmap_colors = [color1, color2]
    cmap = ListedColormap(cmap_colors)

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    reducer = umap.UMAP(random_state=29)
    embedding = reducer.fit_transform(x)

    # 筛选掉x坐标小于0的数据点
    mask = embedding[:, 0] <=100
    embedding_filtered = embedding[mask]
    y_filtered = y[mask]

    np.save('4embedding_filtered.npy', embedding_filtered)
    np.save('4y_filtered.npy', y_filtered)

    plt.scatter(embedding_filtered[:, 0], embedding_filtered[:, 1], c=y_filtered,cmap=cmap, s=10)
    plt.gca().set_aspect('equal', 'datalim')

    c = plt.colorbar(ticks=[0, 1])  # 设置颜色条的刻度位置
    c.ax.tick_params(labelsize=15)
    c.set_ticks([0.25, 0.75])  # 设置刻度为柱子的四分之一和四分之三处
    c.set_ticklabels(['pos', 'neg'])  # 设置刻度标签为 "neg" 和 "pos"
    #
    # del c

    plt.title(f'LSTM', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 保存图像时指定格式为SVG
    plt.savefig(f'../umap_picture/lstm.svg', format='svg', dpi=300)
    plt.show()

def textcnn_draw_umap():

    x = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/TextCNN_embedding.pt")
    y = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/TextCNN_labels.pt")

    cmap_colors = [color1, color2]
    cmap = ListedColormap(cmap_colors)

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    reducer = umap.UMAP(random_state=29)
    embedding = reducer.fit_transform(x)

    mask = embedding[:, 0] >= 0
    embedding_filtered = embedding[mask]
    y_filtered = y[mask]

    np.save('5embedding_filtered.npy', embedding_filtered)
    np.save('5y_filtered.npy', y_filtered)

    plt.scatter(embedding_filtered[:, 0], embedding_filtered[:, 1], c=y_filtered, cmap=cmap, s=10)
    plt.gca().set_aspect('equal', 'datalim')

    c = plt.colorbar(ticks=[0, 1])  # 设置颜色条的刻度位置
    c.ax.tick_params(labelsize=15)
    c.set_ticks([0.25, 0.75])  # 设置刻度为柱子的四分之一和四分之三处
    c.set_ticklabels(['pos', 'neg'])  # 设置刻度标签为 "neg" 和 "pos"
    #
    # del c

    plt.title(f'TextCNN', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 保存图像时指定格式为SVG
    plt.savefig(f'../umap_picture/TextCNN.svg', format='svg', dpi=300)
    plt.show()


def onlyatt_draw_umap():

    x = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/only_att_embeddings.pt")
    y = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/only_att_labels.pt")

    cmap_colors = [color1, color2]
    cmap = ListedColormap(cmap_colors)

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    reducer = umap.UMAP(random_state=29)
    embedding = reducer.fit_transform(x)

    np.save('6embedding_filtered.npy', embedding)
    np.save('6y_filtered.npy', y)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=y,cmap=cmap, s=10)
    plt.gca().set_aspect('equal', 'datalim')

    c = plt.colorbar(ticks=[0, 1])  # 设置颜色条的刻度位置
    c.ax.tick_params(labelsize=15)
    c.set_ticks([0.25, 0.75])  # 设置刻度为柱子的四分之一和四分之三处
    c.set_ticklabels(['pos', 'neg'])  # 设置刻度标签为 "neg" 和 "pos"
    #
    # del c

    plt.title(f'w/o sequence feature', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 保存图像时指定格式为SVG
    plt.savefig(f'../umap_picture/wo_sequence_feature.svg', format='svg', dpi=300)
    plt.show()

def onlyernie_draw_umap():

    x = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/only_ernie_embeddings1.pt")
    y = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/only_ernie_labels1.pt")

    cmap_colors = [color1, color2]
    cmap = ListedColormap(cmap_colors)

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    reducer = umap.UMAP(random_state=29)
    embedding = reducer.fit_transform(x)

    mask = embedding[:, 0] <= 15
    embedding_filtered = embedding[mask]
    y_filtered = y[mask]

    np.save('7embedding_filtered.npy', embedding_filtered)
    np.save('7y_filtered.npy', y_filtered)

    plt.scatter(embedding_filtered[:, 0], embedding_filtered[:, 1], c=y_filtered, cmap=cmap, s=10)


    plt.gca().set_aspect('equal', 'datalim')

    c = plt.colorbar(ticks=[0, 1])  # 设置颜色条的刻度位置
    c.ax.tick_params(labelsize=15)
    c.set_ticks([0.25, 0.75])  # 设置刻度为柱子的四分之一和四分之三处
    c.set_ticklabels(['pos', 'neg'])  # 设置刻度标签为 "neg" 和 "pos"

    plt.title(f'w/o attention', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 保存图像时指定格式为SVG
    plt.savefig(f'../umap_picture/wo_attention.svg', format='svg', dpi=300)
    plt.show()


def mamba_draw_umap():

    x = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/mamba_embedding.pt")
    y = torch.load("/mnt/sdb/home/lrl/code/new_ac4c/ernie/draw/pt/mamba_labels.pt")

    cmap_colors = [color1, color2]
    cmap = ListedColormap(cmap_colors)

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    reducer = umap.UMAP(random_state=29)
    embedding = reducer.fit_transform(x)

    mask = embedding[:, 0] <= 15
    embedding_filtered = embedding[mask]
    y_filtered = y[mask]
    np.save('8embedding_filtered.npy', embedding_filtered)
    np.save('8y_filtered.npy', y_filtered)
    plt.scatter(embedding_filtered[:, 0], embedding_filtered[:, 1], c=y_filtered, cmap=cmap, s=10)


    plt.gca().set_aspect('equal', 'datalim')

    c = plt.colorbar(ticks=[0, 1])  # 设置颜色条的刻度位置
    c.ax.tick_params(labelsize=15)
    c.set_ticks([0.25, 0.75])  # 设置刻度为柱子的四分之一和四分之三处
    c.set_ticklabels(['pos', 'neg'])  # 设置刻度标签为 "neg" 和 "pos"

    plt.title(f'Caduceus_DNA', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 保存图像时指定格式为SVG
    plt.savefig(f'../umap_picture/mamba.svg', format='svg', dpi=300)
    plt.show()

if __name__ == '__main__':
    # ernie_ac4c_draw_umap()
    # transformer_draw_umap()
    # textcnn_draw_umap()
    # gru_draw_umap()
    # lstm_draw_umap()
    # onlyernie_draw_umap()
    # onlyatt_draw_umap()
    mamba_draw_umap()