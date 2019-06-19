from scipy.sparse import vstack
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from tqdm import tqdm_notebook
import numpy as np

# Precompute the phi vector for the NLSA kernel application using batches
# name: the path to the batches
# n_batches: the total number of batches
def precompute_phi_batches(name, n_batches, batch_size):
    res = []
    res.append(1.0)
    for b in tqdm_notebook(range(n_batches)):
        d0 = np.load(name + str(b) + '.npy', mmap_mode='r')
        for i in range(1, batch_size+1):
            if i + (b*batch_size) >= d0.shape[1]:
                continue
            else:
                res.append(d0[i-1, i + (b*batch_size)])
    np.save('phi_vector', res)

# Keeps only the k nearest neighboors from every point
# Computes K-nn by column as batches are split but the matrix is symmetric
# k: the number of nearest neighboors to keep
# name: the path to the batches
# n_batches: the total number of batches
def select_k_closest_batches(name, n_batches, k):
    for b in tqdm_notebook(range(n_batches)):
        d0 = np.load(name + str(b) + '.npy')
        res = np.zeros(d0.shape)
        dim = d0.shape[0]
        for i in range(dim):
            k_indx = sorted(np.argpartition(d0[i], k)[1:k+1])
            res[i, k_indx] = d0[i, k_indx]       
        np.save('k_closest_' + str(b), res)

# Applies the NLSA Kernel and saves the results in batches
# name: the path to the batches
# n_batches: the total number of batches
# b_size: the total size of one batch
# eps: the epsilon factor
def build_nlsa_kernel_batches(name, n_batches, b_size, eps):
    phi_vec = np.load('phi_vector.npy', mmap_mode='r')
    for b in tqdm_notebook(range(n_batches)):
        dist_mat = np.load(name + str(b) + '.npy', mmap_mode='r')
        row, col = dist_mat.shape
        dist_mat = np.power(dist_mat, 2)
        for i in range(row):
            di = phi_vec[i +  (b * b_size)]
            for j in range(col):  
                dj = phi_vec[j]
                dist_mat[i, j] = dist_mat[i, j] / (di*dj)

        indx = np.nonzero(dist_mat)
        dist_mat[indx] = np.exp(-dist_mat[indx]/eps)
        if b == 0:
            np.save('nlsa_kernel_' + str(b), dist_mat[1:, 1:])
        else:
            np.save('nlsa_kernel_' + str(b), dist_mat[:, 1:])
            
# Rebuild the full matrix form batches in sparse format
# name: the path to the batches
# n_batches: the total number of batches
def rebuild_sparse_K(name, n_batches):
    res = lil_matrix([0, 0])
    for b in tqdm_notebook(range(n_batches)):
        dist_mat = np.load(name + str(b) + '.npy')
        if b == 0:
            res = lil_matrix(dist_mat)
        else:
            res = hstack([res, lil_matrix(dist_mat)])
    return res

# Build the sparse Laplace-Beltrami operator
# Uses scipy sparse tricks to keep matrix sparse at all time
# Only use this if memory is not an issue (faster)
# kernel_mat: the rebuild sparse K matrix
# !! MATRIX MUST BE SYMMETRIC
def build_Laplace_Beltrami_operator_sparse_fast(kernel_mat):
    #Equivalent to kernel_mat = csr_matrix(kernel_mat / (kernel_mat.sum(axis=1)[:, None] * kernel_mat.sum(axis=0)[None, :]))
    # This should stay sparse and fit in memory
    d = diags(1/kernel_mat.sum(axis=1).A.ravel())
    e = diags(1/kernel_mat.sum(axis=0).A.ravel())
    kernel_mat = d @ kernel_mat
    kernel_mat = kernel_mat @ e
    # Equivalent to kernel_mat = kernel_mat / kernel_mat.sum(axis=1)
    # This should stay sparse and fit in memory
    c = diags(1/kernel_mat.sum(axis=1).A.ravel())
    kernel_mat = c @ kernel_mat
    # P = I - K
    kernel_mat = identity(kernel_mat.shape[0]) - kernel_mat
    return kernel_mat

# Build the sparse Laplace-Beltrami operator
# Uses scipy sparse tricks to keep matrix sparse at all time
# Use this if memory is an issue (slower)
# Attention: This does not return the usual P matrix (I - K) but only K!!
# Attention: Look for highest eigenvalues with this matrix!!
# kernel_mat: the rebuild sparse K matrix
# !! MATRIX MUST BE SYMMETRIC
def build_Laplace_Beltrami_operator_sparse_slow(kernel_mat):
    #Equivalent to kernel_mat = csr_matrix(kernel_mat / (kernel_mat.sum(axis=1)[:, None] * kernel_mat.sum(axis=0)[None, :]))
    d = kernel_mat.sum(axis=1).ravel()
    e = kernel_mat.sum(axis=0).ravel()
    arr = np.zeros(kernel_mat.data.shape[0])
    for pos, i, j, v in zip(tqdm_notebook(range(kernel_mat.data.shape[0])), kernel_mat.row, kernel_mat.col, kernel_mat.data):
        arr[pos] = v / (d[0, i] * e[0, j])

    # Equivalent to kernel_mat = kernel_mat / kernel_mat.sum(axis=1)
    kernel_mat = coo_matrix((arr, (kernel_mat.row, kernel_mat.col)))
    c = kernel_mat.sum(axis=1).ravel()
    for pos, i, j in zip(tqdm_notebook(range(kernel_mat.data.shape[0])), kernel_mat.row, kernel_mat.col):
        arr[pos] /= c[0, i]

    kernel_mat = coo_matrix((arr, (kernel_mat.row, kernel_mat.col)))
    return kernel_mat