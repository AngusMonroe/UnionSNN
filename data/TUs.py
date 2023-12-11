import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np
import csv
import dgl
import networkx as nx
from dgl.data import TUDataset
from dgl.data import LegacyTUDataset
from tqdm import tqdm
from preprocessing.preprocess import compute_shortest_path
import random
random.seed(42)

from sklearn.model_selection import StratifiedKFold, train_test_split


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in TUsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


class TUsDataset(torch.utils.data.Dataset):
    def __init__(self, name, preprocess=None):
        t0 = time.time()
        self.name = name
        self.max_node_num = 0
        
        #dataset = TUDataset(self.name, hidden_size=1)
        dataset = LegacyTUDataset(self.name, hidden_size=1) # dgl 4.0
        self.input_dim, self.label_dim, self.max_num_node = dataset.statistics()

        # frankenstein has labels 0 and 2; so correcting them as 0 and 1
        if self.name in ["FRANKENSTEIN", "MUTAG"]:
            dataset.graph_labels = np.array([1 if x==2 else x for x in dataset.graph_labels])

        print("[!] Dataset: ", self.name)

        # transfer DGLHeteroGraph to DGLFormDataset
        if preprocess in ['shortest_path_graph']:
            # data = []
            new_adj = os.path.exists('data/TUs/{}_{}_new_adj.npy'.format(self.name, preprocess))
            if new_adj:
                print('Load adj from data/TUs/{}_{}_new_adj.npy'.format(self.name, preprocess))
                adjs = np.load('data/TUs/{}_{}_new_adj.npy'.format(self.name, preprocess), allow_pickle=True)
                adjs = adjs.tolist()
            else:
                print('Feature engineering...')
                adjs = []
            for i in tqdm(range(len(dataset))):
                G, label = dataset[i]
                if G.num_nodes() > self.max_node_num:
                    self.max_node_num = G.num_nodes()
                if not new_adj:
                    new_A = self.update_adj(G, preprocess)
                    adjs.append(new_A)
                else:
                    new_A = torch.tensor(adjs[i])
                e_feat = []
                srcs, dsts = G.edges()
                for j in range(len(list(srcs))):
                    e_feat.append([new_A[srcs[j]][dsts[j]]])
                G.edata['weight'] = torch.tensor((e_feat))

            if not new_adj:
                adjs = np.array(adjs)
                np.save('data/TUs/{}_{}_new_adj.npy'.format(self.name, preprocess), adjs)

        # this function splits data into train/val/test and returns the indices
        self.all_idx = self.get_all_split_idx(dataset)
        
        self.all = dataset
        self.train = [self.format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num in range(10)]
        self.val = [self.format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in range(10)]
        self.test = [self.format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in range(10)]
        
        print("Time taken: {:.4f}s".format(time.time()-t0))

    def get_all_split_idx(self, dataset):
        """
            - Split total number of graphs into 3 (train, val and test) in 80:10:10
            - Stratified split proportionate to original distribution of data with respect to classes
            - Using sklearn to perform the split and then save the indexes
            - Preparing 10 such combinations of indexes split to be used in Graph NNs
            - As with KFold, each of the 10 fold have unique test set.
        """
        root_idx_dir = './data/TUs/'
        if not os.path.exists(root_idx_dir):
            os.makedirs(root_idx_dir)
        all_idx = {}

        # If there are no idx files, do the split and store the files
        if not (os.path.exists(root_idx_dir + dataset.name + '_train.index')):
            print("[!] Splitting the data into train/val/test ...")

            # Using 10-fold cross val to compare with benchmark papers
            k_splits = 10

            cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
            k_data_splits = []

            # this is a temporary index assignment, to be used below for val splitting
            for i in range(len(dataset.graph_lists)):
                dataset[i][0].a = lambda: None
                setattr(dataset[i][0].a, 'index', i)

            for indexes in cross_val_fold.split(dataset.graph_lists, dataset.graph_labels):
                remain_index, test_index = indexes[0], indexes[1]

                remain_set = self.format_dataset([dataset[index] for index in remain_index])

                # Gets final 'train' and 'val'
                train, val, _, __ = train_test_split(remain_set,
                                                     range(len(remain_set.graph_lists)),
                                                     test_size=0.111,
                                                     stratify=remain_set.graph_labels)

                train, val = self.format_dataset(train), self.format_dataset(val)
                test = self.format_dataset([dataset[index] for index in test_index])

                # Extracting only idx
                idx_train = [item[0].a.index for item in train]
                idx_val = [item[0].a.index for item in val]
                idx_test = [item[0].a.index for item in test]

                f_train_w = csv.writer(open(root_idx_dir + dataset.name + '_train.index', 'a+'))
                f_val_w = csv.writer(open(root_idx_dir + dataset.name + '_val.index', 'a+'))
                f_test_w = csv.writer(open(root_idx_dir + dataset.name + '_test.index', 'a+'))

                f_train_w.writerow(idx_train)
                f_val_w.writerow(idx_val)
                f_test_w.writerow(idx_test)

            print("[!] Splitting done!")

        # reading idx from the files
        for section in ['train', 'val', 'test']:
            with open(root_idx_dir + dataset.name + '_'+ section + '.index', 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        return all_idx

    def format_dataset(self, dataset):
        """
            Utility function to recover data,
            INTO-> dgl/pytorch compatible format 
        """
        graphs = [data[0] for data in dataset]
        labels = [data[1] for data in dataset]

        for graph in graphs:
            #graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
            graph.ndata['feat'] = graph.ndata['feat'].float() # dgl 4.0
            # adding edge features for Residual Gated ConvNet, if not there
            if 'feat' not in graph.edata.keys():
                edge_feat_dim = graph.ndata['feat'].shape[1] # dim same as node feature dim
                graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)

        return DGLFormDataset(graphs, labels)
    
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels

    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True
        for split_num in range(10):
            self.train[split_num].graph_lists = [self_loop(g) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [self_loop(g) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [self_loop(g) for g in self.test[split_num].graph_lists]
            
        for split_num in range(10):
            self.train[split_num] = DGLFormDataset(self.train[split_num].graph_lists, self.train[split_num].graph_labels)
            self.val[split_num] = DGLFormDataset(self.val[split_num].graph_lists, self.val[split_num].graph_labels)
            self.test[split_num] = DGLFormDataset(self.test[split_num].graph_lists, self.test[split_num].graph_labels)

    def update_adj(self, G, preprocess):
        A_array = G.adj().to_dense().numpy()
        G = nx.from_numpy_matrix(A_array)

        if preprocess == 'shortest_path_graph':
            weight = compute_shortest_path(A_array, G, graph_type='union_graph')
        else:
            raise NotImplementedError

        w = weight / weight.sum(1, keepdim=True)
        w = torch.nan_to_num(w, nan=0)
        w = w + torch.FloatTensor(A_array)

        return w

    def get_adj_from_weight(self, new_adj):
        weight = new_adj - (new_adj > 0).float()
        weight = weight * (weight > 0).float().sum(1, keepdim=True)

        coeff = new_adj.sum(1, keepdim=True)
        coeff = torch.diag((coeff.T)[0])

        w = new_adj + coeff

        w = w.detach().numpy()
        w = np.nan_to_num(w, nan=0)

        return weight, torch.tensor(w)
