from utils.utils import *

def add_nodes(self, features, adj, labels, idx_train, target_node, n_added=1, n_perturbations=10):
    N = adj.shape[0]
    D = features.shape[1]
    modified_adj = reshape_mx(adj, shape=(N+n_added, N+n_added))
    modified_features = self.reshape_mx(features, shape=(N+n_added, D))

    diff_labels = [l for l in range(labels.max()+1) if l != labels[target_node]]
    diff_labels = np.random.permutation(diff_labels)
    possible_nodes = [x for x in idx_train if labels[x] == diff_labels[0]]

    return modified_adj, modified_features


def generate_injected_features(features, n_added):
    features = features.tolil()
    features[-n_added:] = np.tile(0, (n_added, 1))
    return features


def inserting_nodes(data):
    '''
        inserting nodes to adj, features, and assign labels to the injected nodes
    '''
    adj, features, labels = data.raw_adj, data.features, data.labels
    N = adj.shape[0]
    D = features.shape[1]
    n_added = 500
    print('Number of inserted nodes: %s' % n_added)

    data.adj = reshape_mx(adj, shape=(N+n_added, N+n_added))
    enlarged_features = reshape_mx(features, shape=(N+n_added, D))
    data.features = generate_injected_features(enlarged_features, n_added)

    injected_labels = np.random.choice(labels.max()+1, n_added)


def normalize_adj(adj):
    """
    A_hat = D_rev*(D+A)*D_rev
    A_hat = D_rev*(D-A)*D_rev
    todo: many ways to try!
    """
    A_hat = adj
    degree = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = sp.diags(degree)
    degree_reverse = degree ** (-1 / 2)
    degree_reverse[np.isinf(degree_reverse)] = 0.
    D_hat_reverse = sp.diags(degree_reverse)
    adj = D_hat_reverse * (D_hat + A_hat) * D_hat_reverse
    return adj