import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


class Dataset():

    def __init__(self, data_path, seed=42):
        self.seed = seed
        self.data_path = data_path
        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test(stratify = self.labels)

    def get_train_val_test(self, stratify):
        val_size = 0.25
        test_size = 0.1
        train_size = 1.0 - val_size - test_size

        np.random.seed(self.seed)
        idx = np.arange(len(self.labels))

        # Split train&val / test
        idx_train_and_val, idx_test = train_test_split(idx,
                                                       random_state=self.seed,
                                                       train_size=train_size + val_size,
                                                       stratify=stratify)

        stratify = stratify[idx_train_and_val]

        # Split train / val
        idx_train, idx_val = train_test_split(idx_train_and_val,
                                              random_state=self.seed,
                                              train_size=(train_size / (train_size + val_size)),
                                              stratify=stratify)
        return idx_train, idx_val, idx_test

    def deal_adj(self,adj):
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()
        A_hat = adj
        degree = np.array(np.sum(A_hat, axis=0))[0]
        D_hat = sp.diags(degree)

        degree_reverse = degree ** (-1 / 2)
        degree_reverse[np.isinf(degree_reverse)] = 0.
        D_hat_reverse = sp.diags(degree_reverse)
        adj = D_hat_reverse * (D_hat + A_hat) * D_hat_reverse
        return adj

    def normalize_adj(self, adj):
        A_hat = adj
        degree = np.array(np.sum(A_hat, axis=0))[0]
        D_hat = sp.diags(degree)
        degree_reverse = degree ** (-1 / 2)
        degree_reverse[np.isinf(degree_reverse)] = 0.0
        D_hat_reverse = sp.diags(degree_reverse)
        adj = D_hat_reverse * (D_hat + A_hat) * D_hat_reverse
        return adj

    def load_data(self):
        adj, features, labels = self.load_npz()
        self.raw_adj = adj
        adj = self.deal_adj(adj)

        return adj, features, labels

    def load_npz(self, is_sparse=True):
        adj = np.load(self.data_path+'experimental_adj.pkl', allow_pickle=True)  # [0:543486, 0:543486]
        features = np.load(self.data_path+'experimental_features.pkl', allow_pickle=True)  # [0:543486]
        labels = np.load(self.data_path+'experimental_train.pkl',allow_pickle=True)

        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels
