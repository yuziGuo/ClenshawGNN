import imp
import os
import re
import networkx as nx
import numpy as np
from pandas.core.indexing import need_slice
import scipy.sparse as sp
import scipy
import torch as th
from dgl import DGLGraph
import dgl
from sklearn.model_selection import ShuffleSplit
import pickle as pkl
import sys
from data.loader import loader
from utils.data_split import index_to_mask
import argparse


class nonHomoBench_dataloader(loader):
    def __init__(self, ds_name, device='cuda:0', self_loop=True, 
                    digraph=False, n_cv=3, cv_id=0,
                    needs_edge=False):
        super(nonHomoBench_dataloader, self).__init__(
            ds_name, 
            cross_validation=True, 
            n_cv=n_cv, 
            cv_id=cv_id,
            needs_edge=needs_edge
            )
        self.device = device
        self.digraph = digraph
        self.self_loop = self_loop
        self.root_path = 'dataset/NonHomoBench/'

    def load_vanilla_data(self):
        _ = scipy.io.loadmat(f"{self.root_path}{self.ds_name}.mat")
        _keys = set(_.keys())
  
        # edge
        _edge_index_keys = {'edge_index', 'homo', 'A'}
        _edge_index_key = _edge_index_keys.intersection(_keys).pop()
        _edges = _[_edge_index_key]
        if _edges.shape[0]==_edges.shape[1] and _edges.shape[0]>2:
            _edges = _edges.nonzero()
        edges = th.tensor(_edges, dtype=th.long)
        self.g = dgl.graph((edges[0], edges[1])).int().to(self.device)
        # import ipdb; ipdb.set_trace()
        if self.self_loop:
            self.g = self.g.add_self_loop()

        # node feats
        _feat_keys = {'node_feat', 'features'}
        _feat_key = _feat_keys.intersection(_keys).pop()
        _feats = _[_feat_key]
        if scipy.sparse.isspmatrix(_feats):
            _feats = _feats.todense()
        # self.features = th.FloatTensor(_feats).to(self.device)
        self.features = th.tensor(_feats, dtype=th.float32,device=self.device)
        # label
        _labels = _['label']
        if scipy.sparse.isspmatrix(_labels):
            _labels = _labels.todense()
        self.labels = th.tensor(_labels.flatten(), dtype=th.long, device=self.device)
        
        self.in_feats = self.features.shape[1]
        self.n_classes = self.labels.max().item() + 1
        self.n_edges = self.g.number_of_edges()
        self.n_nodes = self.labels.shape[0]

    def load_a_mask(self, p):
        splits_lst = np.load(f'{self.root_path}splits/{self.ds_name}-splits.npy', allow_pickle=True)
        i = self.cv_id

        for key in splits_lst[i]:
            if not th.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = th.as_tensor(splits_lst[i][key], device=self.device)
        
        train_idxs, val_idxs, test_idxs = splits_lst[i].values()
        self.train_mask = index_to_mask(train_idxs, self.n_nodes).bool()
        self.val_mask = index_to_mask(val_idxs, self.n_nodes).bool()
        self.test_mask = index_to_mask(test_idxs, self.n_nodes).bool()
        return 


def set_args():
    parser = argparse.ArgumentParser(description='Test nonHomoBench_dataloader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--model", type=str, default='GCN', help='[GCN, GCNMV]')
    parser.add_argument("--gpu", type=int, default=1, help="gpu")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--ds-split", type=str, default="standard", help="split by ('standard', 'mg', 'random').")
    parser.add_argument("--self-loop", action='store_true', default=False, help="graph self-loop (default=False)")
    parser.add_argument("--udgraph", action='store_true', default=False, help="undirected graph (default=False)")
    parser.add_argument("--n-cv",  type=int, default=3)
    parser.add_argument("--start-cv",  type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    datasets = [
        # below: 2-class datasets
        # 'pokec',      # DONE
        'YelpChi',    # DONE
        # 'deezer-europe',
        # 'FB100/Penn94',#TODO
        # 'ogbn-proteins'#TODO
        # 'Twitch/DE' # TODO
    ]
    
    for dataset in datasets:
        print(dataset)
        data = nonHomoBench_dataloader(
            dataset, 
            device=args.gpu, 
            self_loop=args.self_loop,
            digraph=not args.udgraph,
            n_cv=args.n_cv, 
            cv_id=args.start_cv,
            needs_edge=True
        )
        print(f'{dataset}: to load')
        success = data.load_data()
        print(f'{dataset} loaded')
        data.load_mask()
        import ipdb; ipdb.set_trace()
        print('---'*10)
