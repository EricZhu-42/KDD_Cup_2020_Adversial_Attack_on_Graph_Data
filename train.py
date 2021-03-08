import random

import torch

from utils.dataset import Dataset
from utils.gcn import GCN
from utils.utils import *

# Basic settings
seed = 2020
use_cuda = True
data_path = r'./dataset/'
model_path = r'./model/model.pth'

# Hidden sizes of GCN, given as a list
hidden_sizes = [64, 32]

# Training iterations
training_iters = 300

# Load path to continue training
load_path = None


# 1 Read and processing data
data = Dataset(data_path=data_path)
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
adj, features, labels = data.adj, data.features, data.labels
print("Data split info train = %d, val = %d, test = %d" % (len(idx_train), len(idx_val), len(idx_test)))

adj = sparse_mx_to_torch_sparse_tensor(adj)
features = sparse_mx_to_torch_sparse_tensor(features)
labels = torch.LongTensor(labels)
device = get_device(use_cuda, "Training")


# 2 Setup GCN model
model = GCN(nfeat=features.shape[1], lr=0.01,nclass=int(labels.max().item())+1,
                   nhid=hidden_sizes, dropout=0.5, weight_decay=5e-4, device=device)

model = model.to(device)
model.fit(features, adj, labels, idx_train, idx_val, normalize=False, train_iters=training_iters, load_path=load_path)

# 3 Save model
torch.save(model.state_dict(), model_path)
