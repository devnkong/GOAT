import sys
import copy
import os.path as osp
from typing import Optional
import numpy as np

import torch
import torch.utils.data
from torch_sparse import SparseTensor, cat

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
import algos


class ClusterData(torch.utils.data.Dataset):
    r"""Clusters/partitions a graph data object into multiple subgraphs, as
    motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
    and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
        save_dir (string, optional): If set, will save the partitioned data to
            the :obj:`save_dir` directory for faster re-use.
            (default: :obj:`None`)
        log (bool, optional): If set to :obj:`False`, will not log any
            progress. (default: :obj:`True`)
    """
    def __init__(self, data, num_parts: int, num_spatial: int, recursive: bool = False,
                 save_dir: Optional[str] = None, log: bool = True):

        assert data.edge_index is not None

        self.num_parts = num_parts
        self.num_spatial = num_spatial

        recursive_str = '_recursive' if recursive else ''
        filename = f'partition_{num_parts}{recursive_str}.pt'
        path = osp.join(save_dir or '', filename)
        if save_dir is not None and osp.exists(path):
            adj, partptr, perm = torch.load(path)
        else:
            if log:  # pragma: no cover
                print('Computing METIS partitioning...', file=sys.stderr)

            N, E = data.num_nodes, data.num_edges
            adj = SparseTensor(
                row=data.edge_index[0], col=data.edge_index[1],
                value=torch.arange(E, device=data.edge_index.device),
                sparse_sizes=(N, N))
            adj, partptr, perm = adj.partition(num_parts, recursive)

            if save_dir is not None:
                torch.save((adj, partptr, perm), path)

            if log:  # pragma: no cover
                print('Done!', file=sys.stderr)

        self.data = self.__permute_data__(data, perm, adj)
        self.partptr = partptr
        self.perm = perm
        self.maxlen = self.get_max_len()


        self.spatial_list = self.save_spatial()
        # filename_spatial = f'partition_{num_parts}{recursive_str}_spatial.pt'
        # path_spatial = osp.join(save_dir or '', filename_spatial)
        # if save_dir is not None and osp.exists(path_spatial):
        #     self.spatial_list = torch.load(path_spatial)
        # else :

    def __permute_data__(self, data, node_idx, adj):
        data = copy.copy(data)
        N = data.num_nodes

        for key, item in data:
            if isinstance(item, torch.Tensor) and item.size(0) == N:
                data[key] = item[node_idx]

        data.edge_index = None
        data.adj = adj

        return data

    def __len__(self):
        return self.partptr.numel() - 1

    def get_spatial(self, N, edge_index) :
        # adj is a dense matrix, if N is too large it would be overflow.
        adj = torch.zeros([N, N], dtype=torch.bool) 
        adj[edge_index[0, :], edge_index[1, :]] = True
        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        spatial = torch.from_numpy((shortest_path_result)).long()
        # num_spatial was set very large, this should be extremely rare
        spatial[spatial >= self.num_spatial] = self.num_spatial-1

        return spatial

    def get_max_len(self) :
        lens = []
        for idx in range(self.__len__()) :
            start = int(self.partptr[idx])
            length = int(self.partptr[idx + 1]) - start
            lens.append(length)
        return max(lens)

    def save_spatial(self):
        spatials = []
        for idx in range(self.__len__()) :
            start = int(self.partptr[idx])
            length = int(self.partptr[idx + 1]) - start

            adj = self.data.adj
            adj = adj.narrow(0, start, length).narrow(1, start, length)
            row, col, _ = adj.coo()
            edge_index = torch.stack([row, col], dim=0)

            spatial = self.get_spatial(length, edge_index)
            spatials.append(spatial)

        return spatials


    def __getitem__(self, idx):
        start = int(self.partptr[idx])
        length = int(self.partptr[idx + 1]) - start

        N, E = self.data.num_nodes, self.data.num_edges
        data = copy.copy(self.data)
        # del data.num_nodes
        adj, data.adj = data.adj, None

        adj = adj.narrow(0, start, length).narrow(1, start, length)
        edge_idx = adj.storage.value()

        for key, item in data:
            if isinstance(item, torch.Tensor) and item.size(0) == N:
                data[key] = item.narrow(0, start, length)
            elif isinstance(item, torch.Tensor) and item.size(0) == E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        row, col, _ = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        data.spatial = self.spatial_list[idx]
        # data.spatial = self.get_spatial(length, data.edge_index)
        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  data={self.data},\n'
                f'  num_parts={self.num_parts}\n'
                f')')


class ClusterDataLoader(torch.utils.data.DataLoader):

    def __init__(self, cluster_dataset, **kwargs):
        self.cluster_dataset = cluster_dataset
        self.maxlen = self.cluster_dataset.maxlen
        super().__init__(torch.arange(len(self.cluster_dataset)), collate_fn=self.__collate__, **kwargs)

    def __collate__(self, indices):

        def pad1d(x, padlen):
            xlen = x.size(0)
            if xlen < padlen:
                new_x = x.new_zeros([padlen])
                new_x[:xlen] = x
                x = new_x
            return x
        
        def pad2d(x, padlen):
            xlen, xdim = x.size()
            if xlen < padlen:
                new_x = x.new_zeros([padlen, xdim])
                new_x[:xlen, :] = x
                x = new_x
            return x

        def padspatial(x, padlen):
            xlen = x.size(0)
            if xlen < padlen:
                new_x = x.new_zeros([padlen, padlen])
                new_x[:xlen, :xlen] = x
                x = new_x
            return x


        xs, spatials, pad_masks, ys = [], [], [], []
        train_masks, valid_masks, test_masks = [], [], []
        for idx in indices :
            
            data = self.cluster_dataset[idx]
            x, spatial, y = data.x,  data.spatial, data.y
            train_mask, valid_mask, test_mask = data.train_mask, data.valid_mask, data.test_mask

            xlen, xdim = x.size()
            pad_mask = ~(torch.arange(self.maxlen) < xlen)     
            pad_masks.append(pad_mask)   

            xs.append(pad2d(x, self.maxlen))
            spatials.append(padspatial(spatial, self.maxlen))
            ys.append(pad2d(y, self.maxlen))

            train_masks.append(pad1d(train_mask, self.maxlen))
            valid_masks.append(pad1d(valid_mask, self.maxlen))
            test_masks.append(pad1d(test_mask, self.maxlen))

        x_batch = torch.stack(xs, dim=0) # (B, maxlen, xdim)
        spatial_batch = torch.stack(spatials, dim=0) # (B, maxlen, maxlen)
        pad_mask_batch = torch.stack(pad_masks, dim=0) # (B, maxlen)
        y_batch = torch.stack(ys, dim=0)

        train_mask_batch = torch.stack(train_masks, dim=0)
        valid_mask_batch = torch.stack(valid_masks, dim=0)
        test_mask_batch = torch.stack(test_masks, dim=0)

        return x_batch, spatial_batch, pad_mask_batch, y_batch, train_mask_batch, valid_mask_batch, test_mask_batch