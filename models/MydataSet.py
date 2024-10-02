from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self,datas,labels):
        self.data = datas
        self.label = labels
    def __getitem__(self, item):
        return self.data[item],self.label[item]
    def __len__(self):
        return len(self.data)