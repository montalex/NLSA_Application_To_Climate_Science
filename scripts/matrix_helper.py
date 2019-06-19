import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from tqdm import tqdm_notebook

n_lat = 72
n_lon = 144

# Flatens a 3D matrix in a vector
# data_mat: the 3D matrix
def mat3_to_vec(data_mat):
    temp_mat_2D = data_mat.reshape(data_mat.shape[0], data_mat.shape[1]*data_mat.shape[2])
    return np.hstack(temp_mat_2D)

# Transforms a vector back in a 3D matrix
# vec: the vector
def vec_to_mat3(vec):
    n_months = int(len(vec) / (n_lat * n_lon))
    temp_mat_2D = np.reshape(vec,(-1, n_lat*n_lon))
    return np.reshape(temp_mat_2D, (n_months, n_lat, n_lon))

# Builds the distance matrix
# data_mat: the data matrix
# q: the chosen time-lag for the embbeding
def build_distance_mat(data_mat, q):
    dim = data_mat.shape[0] - q + 1    
    temp_mat = np.zeros((dim, dim))
    for i in tqdm(range(dim)):
        for j in range(i+1, dim):
            temp_vec_1 = mat3_to_vec(data_mat[i:i+q,])
            temp_vec_2 = mat3_to_vec(data_mat[j:j+q,])
            dist = distance.euclidean(temp_vec_1, temp_vec_2)
            temp_mat[i][j] = dist
            temp_mat[j][i] = dist
            
    return temp_mat

# Precompute full pariwise distances
# data_mat: the data matrix
def precompute_distances(data_mat):
    dim = data_mat.shape[0]
    temp_mat = np.zeros((dim, dim))
    for i in tqdm_notebook(range(dim)):
        for j in range(i+1, dim):
            dist = np.sqrt(np.sum(np.power(data_mat[i] - data_mat[j], 2)))
            temp_mat[i][j] = dist
            temp_mat[j][i] = dist
    return temp_mat

# Builds the distance matrix from a precomputed pariwise distance matrix
# precomputed: the data matrix
# q: the chosen time-lag for the embbeding
def build_distance_mat_precomputed(precomputed, q):
    dim = precomputed.shape[0] - q + 1
    D_mat = np.zeros((dim, dim))
    for i in tqdm_notebook(range(dim)):
        for j in range(i+1, dim):
            vec1 = np.arange(i, i+q)
            vec2 = np.arange(j, j+q)
            dist = np.sqrt(np.sum(np.power(precomputed[vec1, vec2], 2)))
            D_mat[i][j] = dist
            D_mat[j][i] = dist
    return D_mat

# Precompute the phi vector for the NLSA kernel application
# D_mat: the distance matrix
def precompute_phi(D_mat):
    res = []
    res.append(1.0)
    for i in range(1, D_mat.shape[0]):
        res.append(D_mat[i-1, i])
    return res



# Applies a simple gaussian kernel to the distance matrix (sigma = average) 
# dist_mat: the distance matrix
# mult: the gaussian kernel factor
def build_gaussian_kernel_mat(dist_mat, mult):
    temp_mat = np.zeros(dist_mat.shape)
    sigma = mult * np.sum(dist_mat) / np.count_nonzero(dist_mat)
    print(sigma)
    temp_mat[np.nonzero(dist_mat)] = np.exp(-np.power(dist_mat[np.nonzero(dist_mat)], 2) / (2* sigma**2))
    return temp_mat

# Applies the NLSA Kernel
# dist_mat: the distance matrix
# eps: the eps factor
# precomputed_phi: default False, set to true if precomputed phi vector is available
def build_nlsa_kernel(dist_mat, eps, precomputed_phi=False):
    dim = dist_mat.shape[0]
    temp_mat = np.zeros((dim, dim))
    res = np.zeros((dim, dim))
    phi_vec = np.zeros(1)
    if precomputed_phi:
        phi_vec = np.load('dataFull/phi_vector.npy')
    for i in tqdm_notebook(range(1, dim)):
        for j in range(i+1, dim):
            di = 0.0
            dj = 0.0
            if precomputed_phi:
                di = phi_vec[i]
                dj = phi_vec[j]
            else:
                di = dist_mat[i][i-1]
                dj = dist_mat[j][j-1]
                
            d = di*dj
            if d == 0:
                continue
            else:
                temp_mat[i][j] = np.power(dist_mat[i][j], 2) / d
                temp_mat[j][i] = np.power(dist_mat[j][i], 2) / d
    res[np.nonzero(dist_mat)] = np.exp(-temp_mat[np.nonzero(dist_mat)]/eps)
    return res[1:, 1:]

# Builds the Laplace Beltrami operator
# kernel_mat: the K matrix
def build_Laplace_Beltrami_operator(kernel_mat):
    norm = kernel_mat / (kernel_mat.sum(axis=1) * kernel_mat.sum(axis=0))
    P = norm / norm.sum(axis=1) 
    return np.identity(P.shape[0]) - P