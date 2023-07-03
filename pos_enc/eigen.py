import time
import os
import numpy as np
from scipy.sparse.linalg import eigs
from torch_geometric.utils import to_scipy_sparse_matrix

def get_eigen(edge_index, k, name):
    if not os.path.exists('{}_eigenvals{}.npy'.format(name, k)):
        adj = to_scipy_sparse_matrix(edge_index)
        start = time.time()
        eigen_vals, eigen_vecs = eigs(adj.astype(np.float32), k=k, tol=1e-5, ncv=k*3, which='LR') # by default it selects the largest eigen values
        print('Compute eigen: {:.3f} seconds'.format(time.time() - start))
        np.save('{}_eigenvals{}.npy'.format(name, k), eigen_vals)
        np.save('{}_eigenvecs{}.npy'.format(name, k), eigen_vecs)
    else:
        eigen_vals = np.load('{}_eigenvals{}.npy'.format(name, k))
        eigen_vecs = np.load('{}_eigenvecs{}.npy'.format(name, k))
        assert len(eigen_vals) == k
        assert eigen_vecs.shape[1] == k
    return eigen_vals, eigen_vecs


