
from collections import defaultdict
import random
import numpy as np
import scipy.sparse as sp
import torch

class Data():
    def __init__(self, data_path, n_field):
        self.data_path = data_path
        edge_path = data_path + '.edge'
        feature_path = data_path + ".feature"
        label_path = data_path + ".label"
        
        self.n_field = n_field
        self.load_idx()
        print("finish loading idx")

        self.load_edge(edge_path)
        self.load_feature(feature_path)
        self.load_label(label_path, 100)
        self.nfadj = self.load_adj()
    
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx  

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
        
    def load_adj(self):
        edges = np.array(self.edge_list, dtype=np.int32)
        fadj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(self.n_node, self.n_node), dtype=np.float32)
        fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
        nfadj = self.normalize(fadj + sp.eye(fadj.shape[0]))
        nfadj = self.sparse_mx_to_torch_sparse_tensor(nfadj)

        return nfadj 
    
    def load_edge(self, edge_path):
        lines = open(edge_path, 'r').readlines()
        edge_dict = dict()
        self.edge_list = []
        for l in lines:
            tmp = l.strip() #deleting '\n'
            val = [int(i) for i in tmp.split()]
            node_id = val[0] 
            linked_node = val[1:]
            edge_dict[node_id] = linked_node
            for ln in linked_node:
                self.edge_list.append([node_id, ln]) 
    
    def load_feature(self, feature_path):
        lines = open(feature_path, 'r').readlines()
        feature_list = []
        for l in lines:
            tmp = l.strip()
            val = [float(i) for i in tmp.split()]
            feature_list.append(val[1:])

        self.n_node = len(feature_list)
        
        self.features = sp.csr_matrix(np.array(feature_list), dtype=np.float32)
        
        self.features = torch.FloatTensor(np.array(self.features.todense()))


    def load_label(self, label_path, n_ran):
        lines = open(label_path, 'r').readlines()
        self.label_list = []
        for l in lines:
            fl = [0] * self.n_field
            tmp = l.strip()
            val = [int(i) for i in tmp.split()]
            for v in val[1:]:
                fl[v] = 1
            self.label_list.append(fl)

        #self.sample_idx(n_ran)

    def load_idx(self):
        train_path = self.data_path + ".train"
        self.idx_train = []
        lines = open(train_path, 'r').readlines()
        for l in lines:
            self.idx_train.append(int(l))

        test_path = self.data_path + ".test"
        self.idx_test = []
        lines = open(test_path, 'r').readlines()
        for l in lines:
            self.idx_test.append(int(l))
        
        self.idx_train = torch.LongTensor(self.idx_train)
        self.idx_test = torch.LongTensor(self.idx_test)

    def sample_idx(self, n_ran):
        field_dict = defaultdict(list)
        for idx1, field in enumerate(self.label_list):
            for idx2, fie in enumerate(field):
                if fie == 1:
                    field_dict[idx2].append(idx1)

        train_list = set()

        for key, fi in field_dict.items():
            train_list.update(random.sample(fi, n_ran))

        self.idx_train = list(sorted(train_list))
        self.idx_test = list(sorted(set([i for i in range(self.n_node)]).difference(train_list)))

        self.idx_train = torch.LongTensor(self.idx_train)
        self.idx_test = torch.LongTensor(self.idx_test)
