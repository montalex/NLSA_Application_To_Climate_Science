import numpy as np
from tqdm import tqdm

# Computes the distance matrix D using batches
# n_batches: the number of batches composing the full matrix
# q: chosen time-lag for the embbeding

n_batches = 18
q = 1825
for d in range(1):
    print('loading'+str(d))
    d0 = np.load('dataFull/test_D_mat_5000_'+str(d)+'.npy', mmap_mode='r')

    # Load the next batch in order to get the vector necessary for the embbeding
    # Stack it on top of the current batch
    if d+1 < n_batches:
        print('preload' +str(d+1))
        added_vec = np.load('dataFull/test_D_mat_5000_'+str(d+1)+'.npy', mmap_mode='r')
        d0 = np.hstack((d0, added_vec[:, :q]))
        added_vec = None

    dim = d0.shape[0] - q + 1
    size = d0.shape[1] - q
    if d+1 >= n_batches:
        size += 1

    # Compute the distances and store the new batch
    D_mat = np.zeros((dim, size))
    for i in tqdm(range(dim)):
        for j in range(0, size):
            if i == j:
                continue
            else:
                vec1 = np.arange(i, i+q)
                vec2 = np.arange(j, j+q)
                dist = np.sqrt(np.sum(np.power(d0[vec1, vec2], 2)))
                D_mat[i][j] = dist

    np.save('dataFull/D_mat_5000_'+str(d), D_mat)
    D_mat = None
