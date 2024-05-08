from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import torch


class SimpleDataset(Dataset):
    def __init__(self, data, labels_list):
        self.data = data
        self.labels_list = labels_list
        assert self.data.shape[0] == len(self.labels_list), "Image and Mask Patches have different lengths."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sparse_data = self.data[idx]
        label = self.labels_list[idx]
        return sparse_data, label


def get_dataloaders(data, labels, train_idx, val_idx, batch_size=128):
    trainval_set = SimpleDataset(data, labels)
    train_loader = DataLoader(trainval_set, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(trainval_set, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))
    return train_loader, val_loader


def get_data(file_path):
    adata = sc.read(file_path)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(adata.obs["Niche_NMF"].to_numpy())
    labels_map = {label: encoded_label for label, encoded_label in zip(adata.obs["Niche_NMF"].to_numpy(), labels)}
    return torch.tensor(adata.X.toarray(), dtype=torch.float32), labels, labels_map


def get_testloader(file_path, batch_size=256):
    data, labels, labels_map = get_data(file_path)
    test_set = SimpleDataset(data, labels)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader