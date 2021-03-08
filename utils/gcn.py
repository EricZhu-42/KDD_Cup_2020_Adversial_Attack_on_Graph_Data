import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from tqdm import tqdm as tqdm

from utils import utils


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' [' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ']'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device=None, training=True):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"

        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.hidden_sizes = nhid

        assert isinstance(nhid, list) or isinstance(nhid, tuple), "Invalid nhid!"
        self.layer_feature_nums = [self.nfeat] + list(nhid) + [self.nclass]

        # Define layers
        self.layers = list()
        for index in range(len(self.layer_feature_nums) -1):
            self.layers.append(
                GraphConvolution(self.layer_feature_nums[index], self.layer_feature_nums[index+1])
            )
        self.layers = nn.ModuleList(self.layers)

        print("-------- Network Structure --------")
        for layer in self.layers:
            print(layer)
        print("--------------------------------------")

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay

        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def forward(self, x, adj):
        '''
            adj: normalized adjacency matrix
        '''

        for layer in self.layers[:-1]:
            x = layer(x, adj)
            if self.with_relu:
                x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.layers[-1](x, adj)

        return F.log_softmax(x, dim=1)

    def initialize(self):
        for layer in self.layers:
          layer.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, load_path=None):
        '''
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        '''
        self.device = self.layers[0].weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        self._train_with_val(labels, idx_train, idx_val, train_iters, verbose, load_path)

    def save(self, path, best_weight=None, optimizer=None, epoch=0, best_loss_val=424242, best_acc_val=0.0):
        state = {'model': self.state_dict(),
                 'best_weight': best_weight,
                 'optimizer': optimizer,
                 'epoch': epoch,
                 'best_loss_val': best_loss_val,
                 "best_acc_val": best_acc_val}
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(state['model'])


    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose, load_path=None):
        if verbose:
            print('=== training gcn model ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if load_path:
            state = torch.load(load_path)
            self.load_state_dict(state['model'])
            best_weight = state['model']
            best_loss_val = state['best_loss_val']
            best_acc_val = state['best_acc_val']
            start = state['epoch']
            optimizer.load_state_dict(state['optimizer'])
        else:
            best_loss_val = 424242
            best_acc_val = 0
            start = 0

        for i in tqdm(range(train_iters-start), desc="Training"):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                print(best_acc_val)
                self.output = output
                weights = deepcopy(self.state_dict())
            torch.cuda.empty_cache()

            if (i+1) % 25 == 0:
                self.save("./tmp/model_{:d}".format(i+1+start), weights, optimizer.state_dict(), i+1+start, best_loss_val, best_acc_val)

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, idx_test):
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test

    def predict(self, features=None, adj=None):
        '''By default, inputs are unnormalized data'''

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            return self.forward(self.features, adj)

            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)
