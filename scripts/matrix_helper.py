import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from tqdm import tqdm_notebook

n_lat = 72
n_lon = 144

def mat3_to_vec(data_mat):
    temp_mat_2D = data_mat.reshape(data_mat.shape[0], data_mat.shape[1]*data_mat.shape[2])
    return np.hstack(temp_mat_2D)

def vec_to_mat3(vec):
    n_months = int(len(vec) / (n_lat * n_lon))
    temp_mat_2D = np.reshape(vec,(-1, n_lat*n_lon))
    return np.reshape(temp_mat_2D, (n_months, n_lat, n_lon))

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

def build_gaussian_kernel_mat(dist_mat, mult):
    temp_mat = np.zeros(dist_mat.shape)
    sigma = mult * np.sum(dist_mat) / np.count_nonzero(dist_mat)
    print(sigma)
    temp_mat[np.nonzero(dist_mat)] = np.exp(-np.power(dist_mat[np.nonzero(dist_mat)], 2) / (2* sigma**2))
    return temp_mat
    
def build_nlsa_kernel(dist_mat, eps):
    dim = dist_mat.shape[0]
    temp_mat = np.zeros((dim, dim))
    res = np.zeros((dim, dim))
    for i in tqdm_notebook(range(dim)):
        for j in range(i+1, dim):
            di = 1
            if i > 0:
                di = dist_mat[i][i-1]
            dj = dist_mat[j][j-1]
            d = di*dj
            if d == 0:
                d = 1
            temp_mat[i][j] = np.power(dist_mat[i][j], 2) / d
            temp_mat[j][i] = np.power(dist_mat[j][i], 2) / d
    res[np.nonzero(dist_mat)] = np.exp(-temp_mat[np.nonzero(dist_mat)]/eps)
    return res

def build_Laplace_Beltrami_operator(kernel_mat):
    norm = kernel_mat / (kernel_mat.sum(axis=1)[:, None] * kernel_mat.sum(axis=0)[None, :])
    P = norm / norm.sum(axis=1)[:, None]  
    return np.identity(P.shape[0]) - P