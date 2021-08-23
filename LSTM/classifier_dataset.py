import pandas as pd
from torch.utils.data import Dataset


class classifierDataset(Dataset):

    def __init__(self, csv_path, input_len, num_learner, transform=None):
        super(classifierDataset).__init__()
        self.csv_path = csv_path
        self.transform = transform
        self.dataset = pd.read_csv(self.csv_path)
        self.dataset.dropna(axis=0, how='any')
        self.dataset = self.dataset.values
        self.dataset_size = len(self.dataset)
        self.input_len = input_len
        self.num_learner = num_learner
        # print(self.dataset_size)

    def __getitem__(self, index):
        input = self.dataset[index][:self.input_len]
        target = self.dataset[index][self.input_len:self.input_len + self.num_learner]
        return input, target

    def __len__(self):
        return self.dataset_size
