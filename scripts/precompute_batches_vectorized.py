import numpy as np
from tqdm import tqdm

avg = np.load('../avg_MPI_ESM.npy')

batch_size = 10000
n_batches = int(avg.shape[0]/batch_size) + 1

for b in tqdm(range(n_batches)):
    dim = avg.shape[0]
    rest = avg.shape[0] - (b * batch_size)
    size = batch_size
    if size > rest:
        size = rest
    temp_mat = np.sum((avg[None,(b*batch_size):(b*batch_size + size)] - avg[:, None])**2, -1)**0.5
    np.save('../fd_precomputed_dist_'+ str(b), temp_mat)
    print('Saved fd_precomputed_dist_' + str(b))
    
print('All done')
