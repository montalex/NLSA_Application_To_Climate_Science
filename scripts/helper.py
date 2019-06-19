import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import warnings
import io
import base64
from mpl_toolkits.basemap import Basemap
from matplotlib import animation, rc
from IPython.display import HTML

# Import temperature data and returns as numpy array
# path: path to stored data
def import_data(path):
    temp_nc = netCDF4.Dataset(path)
    return np.array(temp_nc.variables['tas'])

# Selects the K closest nearest neighbors
# mat: the distance matrix
# k: the number of nearest neighbors
def select_k_closest(mat, k):
    dim = mat.shape[0]
    temp_mat = np.zeros((dim, dim))
    for i in range(dim):
        min_idx = sorted(np.argpartition(mat[i], k)[1:k+1])
        temp_mat[i][min_idx] = mat[i][min_idx]
        for j in range(k):
            temp_mat[min_idx[j]][i] = mat[min_idx[j]][i]
    return temp_mat

def moving_avg(eigVec, N):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(eigVec, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)
        else:
            moving_ave = cumsum[i]/i
            moving_aves.append(moving_ave)
    return moving_aves

# Builds an animation with the given data using Basemap projection
# mat3d: the data matrix in 3d shape
# f: the frame rate
# !!! Title is hardcoded for a 5y time-lag on full MONTHLY data
def make_animation(mat3d, f, min_max=True):
    fig = plt.figure(figsize=(16,6))
    long = [0, 360]
    lat = [-90, 90]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        map = Basemap(projection = 'cyl', llcrnrlat = lat[0], llcrnrlon = long[0], urcrnrlat = lat[1], urcrnrlon = long[1])
        map.drawcoastlines()
        map.drawparallels( np.arange(-90,90.01,30.0), labels = [1,0,0,0], fontsize = 12, linewidth = 0)
        map.drawmeridians( np.arange(0.,360.,30.), labels = [0,0,0,1], fontsize = 12, linewidth = 0)
        h = map.imshow(mat3d[0,:], cmap = 'jet', interpolation = 'none', animated = True)
        if min_max:
            plt.clim(np.min(mat3d), np.max(mat3d))
        else:
            plt.clim(-np.std(mat3d), np.std(mat3d))
        ax = plt.axes()
        ax.set_title(str(1875))

    def update(m):
        h.set_array(mat3d[m, :])
        ax.set_title(str(1875 + int(m/12)))

    anim = animation.FuncAnimation(fig, update, frames=f, interval=200)
    map.colorbar(h, size = "2%")
    
    return anim

# Save animation as GIF
# anim: the animation
# file_name: the path to store the GIF
# fps: frame per second
def save_as_gif(anim, file_name, fps):
    anim.save(file_name, writer='imagemagick', fps=fps)

# Load the GIF
# file_name: the path to the GIF
def load_gif(file_name):
    video = io.open(file_name, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))

# Save animation as video in mpeg format
# anim: the animation
# file_name: the path to store the video
# fps: frame per second
# !!! IN MOST MACHINE THIS WILL THROW AN ERROR
# !!! NEED MATPLOTLIB 3.1.0
# !!! PATH TO FFMPEG NEEDS TO BE SPECIFIED IN .bashrc
# !!! OR it in the notebook plt.rcParams['animation.ffmpeg_path'] = '/usr/name/path/to/ffmpeg/folder/bin/ffmpeg'
def save_as_vid(anim, file_name, fps):
    writer = animation.FFMpegFileWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(file_name, writer=writer)

# Load the video
# file_name: the path to the video
def load_vid(file_name):
    video = io.open(file_name, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))) 

# Check if a numpy matrix is symmetric
# a: the matrix
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)