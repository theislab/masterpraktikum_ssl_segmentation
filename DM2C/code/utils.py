from __future__ import print_function, absolute_import, division

import os
import pickle
import pdb
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel
from sklearn import metrics
import math
#from sklearn.metrics.cluster.supervised import contingency_matrix
#from munkres import Munkres
from CellPLM.utils import set_seed
from CellPLM_repo.CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
class MFeatDataSet(Dataset):
    '''Mixed-modal feature'''

    def __init__(self, file_mat, has_filename=False):
        self.file_mat = sio.loadmat(file_mat)
        self.lens = len(self.file_mat['X'])
        self.has_filename = has_filename

    def __getitem__(self, index):
        if self.has_filename:
            feat, file, modality = self.file_mat['X'][index]
        else:
            feat, modality = self.file_mat['X'][index]
        feat = feat.squeeze().astype(np.float32)
        cluster_label = self.file_mat['y'][0][index]
        cluster_label = np.float32(cluster_label) - 1
        modality_label = np.float32(modality[0])

        return np.float32(index), feat, modality_label, cluster_label

    def __len__(self):
        return self.lens


class SFeatDataSet(Dataset):
    '''Single modal feature'''

    def __init__(self, file_mat):
        self.file_mat = sio.loadmat(file_mat)
        self.lens = len(self.file_mat['X'])

    def __getitem__(self, index):
        feat = self.file_mat['X'][index][0].squeeze().astype(np.float32)
        return feat

    def __len__(self):
        return self.lens

class h5ad_Dataset(Dataset):
    def __init__(self, h5ad_path):
        self.data = self.prepare_h5ad(h5ad_path)
    def prepare_h5ad(self, h5ad_path):
        data = ad.read_h5ad(h5ad_path)
        data.obs_names_make_unique()
        return data

class img_Dataset(Dataset):
    def __init__(self, img_path):
        names = os.listdir(img_path)
        self.imgs = [img_path+name for name in names]
    def __getitem__(self, index):
        return self.imgs[index]
    def __len__(self):
        return len(self.imgs)

class Custom_Dataloader():
    def __init__(self, dataset, modal, batch_size=32, shuffle=True):
        self.dataset, self.modal, self.batch_size, self.shuffle = dataset, modal, batch_size, shuffle
    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))
        batches = []
        txt_batch, img_batch = [], []
        for index in indices:  # iterate over indices using the iterator
            if self.modal[index] == 0:
                txt_batch.append(self.dataset[index])
            else: img_batch.append(self.dataset[index])
            if len(txt_batch) + len(img_batch) == self.batch_size:
                batches.append([txt_batch, img_batch ])
                yield torch.stack(txt_batch), torch.stack(img_batch)
                txt_batch, img_batch = [], []
        if len(txt_batch) > 0 and len(img_batch) > 0: # return last batch only if both modalities have it
            batches.append([txt_batch, img_batch])
            yield torch.stack(txt_batch), torch.stack(img_batch)
    def __len__(self):
        dataset_length = len(self.dataset)
        batch_number = dataset_length / self.batch_size
        length = math.ceil(batch_number)
        return length

def run_PCA_on_modal(x, n_feat):
    ''' have to wait for bigger dataset with this
    x = x.detach().numpy()
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_feat)
    x = pca.fit_transform(x)
    return torch.tensor(x)
    '''
    return x[:, :n_feat] # for no only take the first 512 to match the dimensions

def best_map(L1, L2):
    # L1 should be the ground-truth labels and L2 should be the clustering labels we got
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
    return metrics.normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')


def get_fpr(y_true, y_pred):
    n_samples = np.shape(y_true)[0]
    c = contingency_matrix(y_true, y_pred, sparse=True)
    tk = np.dot(c.data, np.transpose(c.data)) - n_samples  # TP
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples  # TP+FP
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples  # TP+FN
    precision = 1. * tk / pk if tk != 0. else 0.
    recall = 1. * tk / qk if tk != 0. else 0.
    f = 2 * precision * recall / (precision + recall) if (precision +
                                                          recall) != 0. else 0.
    return f, precision, recall


def get_purity(y_true, y_pred):
    c = metrics.confusion_matrix(y_true, y_pred)
    return 1. * c.max(axis=0).sum() / np.shape(y_true)[0]


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