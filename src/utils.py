from __future__ import print_function, absolute_import, division

import os
import math
import anndata as ad
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import contingency_matrix
from munkres import Munkres


class h5ad_Dataset(Dataset):
    def __init__(self, h5ad_path):
        self.data = self.prepare_h5ad(h5ad_path)

    def prepare_h5ad(self, h5ad_path):
        data = ad.read_h5ad(h5ad_path)
        data.obs_names_make_unique()
        return data


class img_Dataset(Dataset):
    def __init__(self, img_path):
        self.imgs = self._get_image_paths(img_path)

    def _get_image_paths(self, root_dir):
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.tif')):  # Add more extensions if needed
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class Custom_Dataloader:
    def __init__(self, dataset, modal, batch_size=32, shuffle=True):
        self.dataset, self.modal, self.shuffle = dataset, modal, shuffle
        self.batch_size, self.txt_idx, self.img_idx = self._get_indices(
            batch_size
        )  # update batch_size and get indices

    def _get_indices(self, batch_size):
        # get indices for both modalities
        img_idx = np.where(np.array(self.modal) == 1)[0]
        txt_idx = np.where(np.array(self.modal) == 0)[0]
        if self.shuffle:
            txt_idx = np.random.permutation(txt_idx)
            img_idx = np.random.permutation(img_idx)
        # get batch size depending on embedding availability
        updated_batch_size = min(int(batch_size / 2), len(txt_idx), len(img_idx))
        return updated_batch_size * 2, txt_idx, img_idx

    def __iter__(self):
        indices = min(len(self.img_idx), len(self.txt_idx))
        txt_batch, img_batch = [], []
        for index in range(indices):  # iterate over indices using the iterator
            txt_batch.append(self.dataset[self.txt_idx[index]])
            img_batch.append(self.dataset[self.img_idx[index]])
            if len(txt_batch) + len(img_batch) == self.batch_size:
                yield torch.stack(txt_batch), torch.stack(img_batch)
                txt_batch, img_batch = [], []
        if (
            len(txt_batch) > 0
            and len(img_batch) > 0
            and len(txt_batch) == len(img_batch)
        ):  # returning last batch with size < batch_size
            yield torch.stack(txt_batch), torch.stack(img_batch)

    def __len__(self):
        dataset_length = len(self.dataset)
        batch_number = dataset_length / self.batch_size
        length = math.ceil(batch_number)
        return length


def run_PCA(x, n_components):
    x = x.detach().cpu().numpy()
    # scaling is primarily important if the scales of the features differ
    # x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_components)
    x = pca.fit_transform(x)
    return torch.tensor(x)


def best_map(L1, L2):
    # L1 should be the ground-truth labels and L2 should be the clustering labels
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = (L1 == Label1[i]).astype(float)
        for j in range(nClass2):
            ind_cla2 = (L2 == Label2[j]).astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def get_ar(y_true, y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)


def get_nmi(y_true, y_pred):
    return metrics.normalized_mutual_info_score(
        y_true, y_pred, average_method="arithmetic"
    )


def get_fpr(y_true, y_pred):
    n_samples = np.shape(y_true)[0]
    c = contingency_matrix(y_true, y_pred, sparse=True)
    tk = np.dot(c.data, np.transpose(c.data)) - n_samples  # TP
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples  # TP+FP
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples  # TP+FN
    precision = 1.0 * tk / pk if tk != 0.0 else 0.0
    recall = 1.0 * tk / qk if tk != 0.0 else 0.0
    f = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0.0
        else 0.0
    )
    return f, precision, recall


def get_purity(y_true, y_pred):
    c = metrics.confusion_matrix(y_true, y_pred)
    return 1.0 * c.max(axis=0).sum() / np.shape(y_true)[0]


def calculate_metrics(y, y_pred):
    y_new = best_map(y, y_pred)
    acc = metrics.accuracy_score(y, y_new)
    ar = get_ar(y, y_pred)
    nmi = get_nmi(y, y_pred)
    f, p, r = get_fpr(y, y_pred)
    purity = get_purity(y, y_pred)
    return acc, ar, nmi, f, p, r, purity


def check_dir_exist(dir_):
    if not os.path.isdir(dir_):
        os.mkdir(dir_)
