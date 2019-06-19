import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

avg = np.load('../avg_MPI_ESM.npy')
avg = avg.reshape((avg.shape[0], avg.shape[1]*avg.shape[2]))

batch_size = 10000
n_batches = int(avg.shape[0]/batch_size) + 1

for b in range(n_batches):
    dim = avg.shape[0]
    rest = avg.shape[0] - (b * batch_size)
    size = batch_size
    if size > rest:
        size = rest
    temp_mat = np.zeros((dim, size))
    for i in tqdm(range(dim)):
        for j in range(size):
            dist = distance.euclidean(avg[i], avg[(b * batch_size) + j])
            temp_mat[i][j] = dist
            
    np.save('../fd_precomputed_dist_'+ str(b), temp_mat)
    print('Saved fd_precomputed_dist_' + str(b))
    
print('All done')