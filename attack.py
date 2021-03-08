import copy
import pickle

from utils.dataset import Dataset
from utils.gcn import GCN
from utils.poison import *
from utils.utils import *

# Basic settings
seed = 2020
use_cuda = True
data_path = r'./dataset/'
model_path = r'./model/model_64_32_best.pth'

# Hidden sizes of GCN, given as a list
hidden_sizes = [64, 32]

# Attack iterations
attack_iters = 10

# Max value of feature
max_delta = 3.0

# How the new nodes are connected
# Options: "seq", "same", "diff"
connect_mode = "seq"


# 1 Read and processing data
data = Dataset(data_path=data_path)
injecting_nodes(data)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


# 2 Load the surrogate model
device = get_device(use_cuda, "Attacking")
surrogate = GCN(nfeat=100, lr=0.01,nclass=labels.max().item()+1,
                   nhid=hidden_sizes, dropout=0.5, weight_decay=5e-4, device=device, training=False)
surrogate.load_state_dict(torch.load(model_path))
surrogate = surrogate.to(device)


# 3 Create graph for classification
start = 543486
idx_opt = np.array(range(start, start+50000))  # the candidate nodes to connect

ori_adj = copy.deepcopy(adj)
adj = adj.tolil()
adj.setdiag(0)
for i in range(500):
    node_connection = list(range(start+i*100,start+(i+1)*100))
    for item in node_connection:
        adj[593486+i, item] = 1
        adj[item, 593486 + i] = 1
adj = adj.astype("float32").tocsr()


# 4 Processing features. Making features to be requiring grad
adj_norm = normalize_adj(adj)
adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
labels = torch.LongTensor(labels).to(device)
features = features.todense()
features = torch.FloatTensor(features).to(device)
features.requires_grad_(True)


# 5 Predict on the raw data, use predicted label for test set (last 50000 nodes)
raw_output = surrogate.predict(features, adj_norm)
raw_output = raw_output.max(1)[1]


# 6 Modify structure based on predicted test set labels
all_list = list()
for i in range(18):
    all_list.append(list())
for i in range(50000):
    all_list[raw_output[start + i]].append(start + i)

idx_opt_list = list()
if connect_mode == 'seq':
    # seq: new_node[0] <-> test_nodes[0:100], etc.
    idx_opt_list = np.array(range(start, start+50000))
elif connect_mode == 'same':
    # same: test_nodes with same labels are connected to the same new_node
    for lst in all_list:
        idx_opt_list.extend(lst)
elif connect_mode == 'diff':
    # same: test_nodes with same labels are connected to different new_nodes
    while True:
        modified = False
        for lst in all_list:
            if len(lst) > 0:
                idx_opt_list.append(lst.pop())
                modified = True
        if not modified:
            break
else:
    assert False, "Invalid connect mode"
assert len(idx_opt_list) == 50000

adj = ori_adj.tolil()
adj.setdiag(0)
for i in range(500):
    node_connection = idx_opt_list[100*i : 100*i+100]
    for item in node_connection:
        adj[593486+i, item] = 1
        adj[item, 593486 + i] = 1
adj = adj.astype("float32").tocsr()
adj_norm = normalize_adj(adj)
adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)


# 7 attacking on the features
iteration = attack_iters
factor = 5e5
while iteration > 0:
    output = surrogate.predict(features, adj_norm)
    loss_test = F.cross_entropy(output[idx_opt], raw_output[idx_opt])
    acc_test = accuracy(output[idx_opt], raw_output[idx_opt])
    grad = torch.autograd.grad(loss_test, features, retain_graph=True)[0]
    for k in range(500):
        line = 593486 + k
        for n in range(100):
            if grad[line][n]>0:
                features[line, n] = min(max_delta, features[line, n] + factor  * grad[line][n])
            elif grad[line][n]<0:
                features[line, n] = max(-max_delta, features[line, n] + factor * grad[line][n])
    print("Loss= %.4f, accuracy= %.4f}" % (loss_test.item(), acc_test.item()))
    iteration -=1


# 8 Save result
adj = adj.astype("float32").tocsr()
features = features.cpu().detach().numpy()
np.save('./output/feature.npy',features[593486:])
with open('./output/adj.pkl','wb') as f:
    pickle.dump(adj[593486:,:],f)