from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
import torch_sparse
from torch_sparse import SparseTensor

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
import numpy as np

import os
import time

import pdb

dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M 
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M 
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}

def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding
    
    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
    
    return label, features

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx

def even_quantile_labels(vals, nclasses=5, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class LocalSampler(torch.utils.data.DataLoader):
    def __init__(self, edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, **kwargs):

        edge_index = edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        # Save for Pytorch Lightning...
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1

            value = torch.arange(edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        batch_size: int = len(batch)

        edge_index, edge_dist = [], []
        for i in range(len(batch)) :
            out = self.sample_one(batch[i:i+1])
            edge_index.append(out[0])
            edge_dist.append(out[1])
        edge_index = torch.cat(edge_index, dim=1)
        edge_dist = torch.cat(edge_dist, dim=1)

        node_idx = torch.unique(edge_index[0]) # source nodes, will include target
        node_idx_flag = torch.tensor([i not in batch for i in node_idx])
        node_idx = node_idx[node_idx_flag]
        node_idx = torch.cat([batch, node_idx])

        # relabel
        node_idx_all = torch.zeros(self.num_nodes, dtype=torch.long)
        node_idx_all[node_idx] = torch.arange(node_idx.size(0))
        edge_index = node_idx_all[edge_index]

        return torch.cat([edge_index, edge_dist], dim=0), node_idx, batch_size

    def sample_one(self, idx):
        assert idx.dim() == 1 and len(idx) == 1

        n_id = idx
        ptrs = []
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            # e_id = adj_t.storage.value()
            total_target = adj_t.sparse_sizes()[::-1] # (total, target)
            total_size = total_target[0]
            ptrs.append(total_size)

        target = torch.tensor([idx.item()] * len(n_id))
        dist = torch.ones(len(n_id))

        for i, ptr in enumerate(reversed(ptrs)) :
            dist[:ptr] = len(self.sizes) - i
        dist[0] = 0
        edge_dist = dist.long()
        # edge_index = torch.stack([target, n_id]) #BUG
        edge_index = torch.stack([n_id, target]) #edge_index[0]:source, edge_index[1]:target
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html

        edge_dist_count = [ptrs[i+1]-ptrs[i] for i in range(len(ptrs)-1)]
        edge_dist_count = [1, ptrs[0]-1] + edge_dist_count
        edge_dist_count = torch.tensor(edge_dist_count)
        edge_dist = torch.stack([edge_dist, edge_dist_count[edge_dist]])

        return edge_index, edge_dist

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'


class LocalSamplerNew(torch.utils.data.DataLoader):
    def __init__(self, edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, load=None, **kwargs):

        edge_index = edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        # Save for Pytorch Lightning...
        self.edge_index = edge_index
        self.node_idx = node_idx

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.__val__ = None
        self.load = load

        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        assert not self.is_sparse_tensor
        if (num_nodes is None and node_idx is not None
                and node_idx.dtype == torch.bool):
            num_nodes = node_idx.size(0)
        if (num_nodes is None and node_idx is not None
                and node_idx.dtype == torch.long):
            num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
        if num_nodes is None:
            num_nodes = int(edge_index.max()) + 1
        self.num_nodes = num_nodes


        if self.load is not None :

            if os.path.isfile(self.load) :
                self.adjs_t = torch.load(self.load, map_location=torch.device('cpu'))
            else :
                edge_index_list = self.process_hop_adjs()
                self.adjs_t = []
                for eidx in edge_index_list :

                    value = torch.arange(eidx.size(1)) if return_e_id else None
                    adj_t = SparseTensor(row=eidx[0], col=eidx[1], value=value,
                                            sparse_sizes=(num_nodes, num_nodes)).t()
                    adj_t.storage.rowptr()
                    self.adjs_t.append(adj_t)

                torch.save(self.adjs_t, self.load)

        else :
            raise NotImplementedError

        if node_idx is None:
            node_idx = torch.arange(self.adjs_t[0].sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def process_hop_adjs(self) :
        
        edge_index = self.edge_index
        edge_index_list = [edge_index]                
        N = self.num_nodes
        edge_index_tmp = edge_index

        for _ in  range(len(self.sizes)-1) :

            edge_index2, _ = torch_sparse.spspmm(edge_index, torch.ones([edge_index[0].size(0)]), edge_index_tmp, torch.ones([edge_index_tmp[0].size(0)]), N, N, N)

            idx = edge_index_tmp[0] * N + edge_index_tmp[1]
            idx2 = edge_index2[0] * N + edge_index2[1]
            mask = torch.from_numpy(np.isin(idx2.cpu().numpy(), idx.cpu().numpy()))
            mask = ~mask  # Invert mask to only contain the elements not in `idx`
            edge_index2 = edge_index2[:, mask]

            edge_index_list.append(edge_index2)
            edge_index_tmp = edge_index2

            pdb.set_trace()


        return edge_index_list


    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        batch_size: int = len(batch)

        edge_index, edge_dist = [], []

        for i, size in enumerate(self.sizes) :
            adj_t, n_id = self.adjs_t[i].sample_adj(batch, size, replace=False)
            tgt, src, _ = adj_t.coo()
            edge_index.append(torch.stack([src, tgt], dim=0))
            edge_dist.append( torch.stack( [torch.ones(src.shape[0]) * (i+1), torch.ones(src.shape[0]) * size ], dim=0 ))

        self_loop = torch.stack([batch, batch], dim=0)
        edge_index.append(self_loop)
        self_edge_dist = torch.stack([torch.zeros(batch_size, dtype=torch.int), torch.ones(batch_size)], dim=0)
        edge_dist.append(self_edge_dist)

        edge_index = torch.cat(edge_index, dim=1)
        edge_dist = torch.cat(edge_dist, dim=1)

        node_idx = torch.unique(edge_index[0]) # source nodes, will include target
        node_idx_flag = torch.tensor([i not in batch for i in node_idx])
        node_idx = node_idx[node_idx_flag]
        node_idx = torch.cat([batch, node_idx])

        # relabel
        node_idx_all = torch.zeros(self.num_nodes, dtype=torch.long)
        node_idx_all[node_idx] = torch.arange(node_idx.size(0))
        edge_index = node_idx_all[edge_index]

        return torch.cat([edge_index, edge_dist], dim=0), node_idx, batch_size

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'

if __name__ == '__main__' :
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/cmlscratch/kong/datasets/ogb')
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    
    # train_loader = LocalSampler(data.edge_index, node_idx=None, num_nodes=data.num_nodes,
    #                         sizes=[20,10,5], batch_size=4096,
    #                         shuffle=True, num_workers=0, drop_last=True) # total time: 114.93208193778992
    # train_loader = LocalSampler(data.edge_index, node_idx=None, num_nodes=data.num_nodes,
    #                         sizes=[20,10,5], batch_size=4096,
    #                         shuffle=True, num_workers=16, drop_last=True) # total time: 14.570346355438232
    # train_loader = LocalSampler(data.edge_index, node_idx=None, num_nodes=data.num_nodes,
    #                         sizes=[20,10,5], batch_size=4096,
    #                         shuffle=True, num_workers=8, drop_last=True) # total time: 26.613992929458618
    # train_loader = LocalSampler(data.edge_index, node_idx=None, num_nodes=data.num_nodes,
    #                         sizes=[20, 5], batch_size=4096,
    #                         shuffle=True, num_workers=16, drop_last=True) # total time: 8.636133432388306
    # train_loader = LocalSampler(data.edge_index, node_idx=None, num_nodes=data.num_nodes,
    #                         sizes=[20, 5], batch_size=10000,
    #                         shuffle=True, num_workers=16, drop_last=True) # total time: 6.438913345336914

    # train_loader = LocalSamplerNew(data.edge_index, node_idx=None, num_nodes=data.num_nodes,
    #                         sizes=[20, 200, 1000], batch_size=4096,
    #                         shuffle=True, num_workers=0, drop_last=True, load='arxiv_adjs_t.pt')  # total time:135.78877925872803
    # train_loader = LocalSamplerNew(data.edge_index, node_idx=None, num_nodes=data.num_nodes,
    #                         sizes=[20,100,500], batch_size=4096,
    #                         shuffle=True, num_workers=0, drop_last=True, load='arxiv_adjs_t.pt')  #total time:112.15710496902466
    # train_loader = LocalSamplerNew(data.edge_index, node_idx=None, num_nodes=data.num_nodes,
    #                         sizes=[20,100,500], batch_size=4096,
    #                         shuffle=True, num_workers=8, drop_last=True, load='arxiv_adjs_t.pt')  #total time: total time: 19.727849006652832
    # train_loader = LocalSamplerNew(data.edge_index, node_idx=None, num_nodes=data.num_nodes,
    #                         sizes=[20,100,500], batch_size=4096,
    #                         shuffle=True, num_workers=16, drop_last=True, load='arxiv_adjs_t.pt')  #total time: total time: 12.428156614303589

    train_loader = LocalSampler(data.edge_index, node_idx=None, num_nodes=data.num_nodes,
                            sizes=[20,5], batch_size=1024, num_workers=4) # total time: 6.438913345336914

    start_time = time.time()
    for i, (edge_index, node_idx, bs) in enumerate(train_loader) :
        print(i)
        pass
    end_time = time.time()
    
    print('total time:')
    print(end_time - start_time)
    