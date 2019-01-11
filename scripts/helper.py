import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import warnings
import io
import base64
from mpl_toolkits.basemap import Basemap
from matplotlib import animation, rc
from IPython.display import HTML

def import_data(path):
    temp_nc = netCDF4.Dataset(path)
    return np.array(temp_nc.variables['tas'])

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

def plot_map(month_idx):
    fig = plt.figure(figsize=(16,6))
    long = [0, 360]
    lat = [-90, 90]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        map = Basemap(projection = 'cyl', llcrnrlat = lat[0], llcrnrlon = long[0], urcrnrlat = lat[1], urcrnrlon = long[1])
        map.drawcoastlines()
        map.drawparallels( np.arange(-90,90.01,30.0), labels = [1,0,0,0], fontsize = 12, linewidth = 0)
        map.drawmeridians( np.arange(0.,360.,30.), labels = [0,0,0,1], fontsize = 12, linewidth = 0)

        h = map.imshow(temp_ncdata[month_idx,:], cmap = 'jet', interpolation = 'none', animated = True)
        map.colorbar(h, size = "2%")
        plt.title("Near-surface air temperature")
        
def make_animation(mat3d, f):
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

    def update(m):
        h.set_array(mat3d[m, :])

    anim = animation.FuncAnimation(fig, update, frames=f, interval=200)#, blit=True)
    map.colorbar(h, size = "2%")#, pad="40%", ticks = range(cmin, cmax + 1, cint))
    
    return anim

def save_as_gif(anim, file_name, fps):
    anim.save(file_name, writer='imagemagick', fps=fps)
    
def load_gif(file_name):
    video = io.open(file_name, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))

def save_as_vid(anim, file_name):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(file_name, writer=writer)
    
def load_vid(file_name):
    video = io.open(file_name, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))) 
    