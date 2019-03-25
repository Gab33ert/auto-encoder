# topographique auto encoder sturcture
import os
import tqdm
import voronoi
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation


def connect(Pl,l):
    sigma=300
    W=[]
    for j in range(l-1):
        P1=Pl[j]
        P2=Pl[j+1]
        n1 = len(P1)
        n2 = len(P2)
    
        dP = P1.reshape(1,n1,2) - P2.reshape(n2,1,2)
    
            
    
        # Distances
    
        #D = np.hypot(dP[...,0], dP[...,1])
        D = dP[...,1]
    
        w = np.zeros((n2,n1))
        for i in range(n1):
                for j in range(n2):
                    if (np.random.uniform(0,1) < np.exp(-(D[j,i]**2)/(2*sigma**2))):
                        w[j,i]=1
        W.append(w)
    
    return W

def one_step_forward(W,x):
    return 1/(1+np.exp(-W.dot(np.transpose(x))))

def forward(x,W):
    l=len(W)
    for i in range(l):
        x=one_step_forward(W[i],x) 
    return x


def split(P, l):#l number of layer 
    n=len(P)
    layer=[]
    for j in range(l):
        temp=[]
        c=1
        for i in range(n):
            if((1000/l)*j<P[i,0]<=(1000/l)*(j+1)):
                if(c==0):
                    temp.append(P[i,:])
                else:
                    #layer.append(P[i,:])
                    temp.append(P[i,:])
                    c=0
        temp=np.array(temp)
        layer.append(temp)
    return np.array(layer)

def build(n_cells=1000, n_input_cells = 32, n_output_cells = 32,
          n_input = 3, n_output = 3, sparsity = 0.01, seed=0,l=10):
    """

    Parameters:
    -----------

    n_cells:        Number of cells in the reservoir
    n_input_cells:  Number of cells receiving external input
    n_output_cells: Number of cells sending external output
    n_input:        Number of external input
    n_output:       Number of external output
    sparsity:       Connection rate
    seed:           Seed for the random genrator

    
    """
    
    np.random.seed(seed)
    density    = np.ones((1000,1000))
    n=1000
    for i in range(n):
	    ii=i/(n-1)
	    density[:,i]=((3)*ii*(ii-1)+1)*np.ones((1,n)) #neurone density
    density_P  = density.cumsum(axis=1)
    density_Q  = density_P.cumsum(axis=1)
    filename = "aha.npy"#"CVT-%d-seed-%d.npy" % (n_cells,seed)

    if not os.path.exists(filename):
        cells_pos = np.zeros((n_cells,2))
        cells_pos[:,0] = np.random.uniform(0, 1000, n_cells)
        cells_pos[:,1] = np.random.uniform(0, 1000, n_cells)
        for i in tqdm.trange(75):
            _, cells_pos = voronoi.centroids(cells_pos, density, density_P, density_Q)
        np.save(filename, cells_pos)

    cells_pos = np.load(filename)
    cells_pos=split(cells_pos,l)
  
    #X,Y = cells_pos[:,0], cells_pos[:,1]

    
    W=connect(cells_pos, l)
    return cells_pos/1000, W#, W_in, W_out, bias
    

# Build
# ------
#P, W, W_in, W_out, bias = build(1000, 32, 32, n_input=1, n_output=1,sparsity=0.05, seed=0)
 
P, W = build(20, 32, 32, n_input=1, n_output=1,sparsity=0.05, seed=0,l=3)

print(P, W)

x=np.array([1,2,3])
#print(forward(x,W))


'''
def connect(P, k=10):
    n = len(P)
    dP = P.reshape(1,n,2) - P.reshape(n,1,2)
        
    # Distances
    D = np.hypot(dP[...,0], dP[...,1])

    # k nearest neighbors
    # I = np.argsort(D, axis=1)
    # W = np.zeros((n,n))
    #for i in range(n):
    #    # Connections (no self-connection -> 1:k+1)
    #    W[i,I[i,1:k+1]] = 1


    # Isotropic connections
    # W = np.zeros((n,n))
    # for i in range(n):
    #     R = D[i]/1000.0
    #     W[i] = np.random.uniform(0,1,len(R)) < np.exp(-R/0.125)
    #     W[i,i] = 0

    # Non-isotropic connections
    I = np.argsort(D, axis=1)
    A = np.zeros((n,n))
    for i in range(n):
        A[i] = np.arctan2(dP[i,:,1], dP[i,:,0]) * 180.0 / np.pi

    n = len(P)
    W = np.zeros((n,n))
    for i in range(n):
        p = 0
        for j in range(1,n):
            if  A[i,I[i,j]] > 90 or A[i,I[i,j]] < -90: 
            # if -135 < A[i,I[i,j]] < -45:
                if np.random.uniform(0,1) < 0.25:
                    W[I[i,j],i] = 1
                p += 1
            if p > k:
                break

    return W


def build(n_cells=1000, n_input_cells = 32, n_output_cells = 32,
          n_input = 3, n_output = 3, sparsity = 0.01, seed=0):
    """

    Parameters:
    -----------

    n_cells:        Number of cells in the reservoir
    n_input_cells:  Number of cells receiving external input
    n_output_cells: Number of cells sending external output
    n_input:        Number of external input
    n_output:       Number of external output
    sparsity:       Connection rate
    seed:           Seed for the random genrator

    
    """
    
    np.random.seed(seed)
    density    = np.ones((1000,1000))
    density_P  = density.cumsum(axis=1)
    density_Q  = density_P.cumsum(axis=1)
    filename = "CVT-%d-seed-%d.npy" % (n_cells,seed)

    if not os.path.exists(filename):
        cells_pos = np.zeros((n_cells,2))
        cells_pos[:,0] = np.random.uniform(0, 1000, n_cells)
        cells_pos[:,1] = np.random.uniform(0, 1000, n_cells)
        for i in tqdm.trange(75):
            _, cells_pos = voronoi.centroids(cells_pos, density, density_P, density_Q)
        np.save(filename, cells_pos)

    cells_pos = np.load(filename)
    X,Y = cells_pos[:,0], cells_pos[:,1]
    cells_in  = np.argpartition(X, +n_input_cells)[:+n_input_cells]
    cells_out = np.argpartition(X, -n_output_cells)[-n_output_cells:]

    k = int(n_cells * sparsity)
    W = connect(cells_pos, k=k)

    W_in  = np.zeros((n_input, n_cells))
    W_in[:,cells_in] = 1

    W_out = np.zeros((n_cells, n_output))
    W_out[cells_out,:] = 1

    # Bias unit index (arbitrarily set to the one nearest from (0,0))
    bias = np.argmin(np.hypot(X,Y))

    W[:,bias] = 1
    W_out[bias,:] = 1
    
    return cells_pos/1000, W, W_in, W_out, bias
    

# Build
# ------
P, W, W_in, W_out, bias = build(1000, 32, 32, n_input=1, n_output=1,
                                              sparsity=0.05, seed=0)
leak = 0.01
radius = 1.0
# ------


# Build weights (taking bias into account)
W_in  *= np.random.uniform(-1.0, 1.0, W_in.shape)
W     *= np.random.uniform(-1.0, 1.0, W.shape)
W     *= radius / np.abs(np.linalg.eig(W)[0]).max()
W_out *= np.random.uniform(-1.0, 1.0, W_out.shape)


sample_input  = .5
V = np.tanh(np.dot(sample_input, W_in))
V[0,bias] = 1.0 
#for i in range(25):
#    sample_input = .5
#    V = (1-leak)*V + leak*np.tanh(np.dot(sample_input, W_in) + np.dot(V,W))
#    V[0,bias] = 1.0 
 #    output = np.dot(V,W_out)


facecolors = np.ones((V.size,4))
cmap = plt.get_cmap('viridis') 
norm = matplotlib.colors.Normalize(vmin=-1, vmax=+1)
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
cmap._A = []
facecolors[:] = cmap.to_rgba(V)


# Display
fig = plt.figure(figsize=(6,6))

ax = plt.subplot(1, 1, 1, aspect=1)
patches = []
regions, vertices = voronoi.voronoi_finite_polygons_2d(P)
for i,region in enumerate(regions):
    patches.append(Polygon(vertices[region]))
collection = PatchCollection(patches,
                             facecolor=facecolors, edgecolor="black", linewidth=0.25)
ax.add_collection(collection)
ax.scatter(P[bias,0], P[bias,1], s=10,
           facecolor="black", edgecolor="black", linewidth=.5)

X, Y = P[:,0], P[:,1]
ax.set_xlim(0,1), ax.set_xticks([])
ax.set_ylim(0,1), ax.set_yticks([])

def update(frame_number):
    global V

    sample_input =  2*np.cos(frame_number/50)

    for i in range(25):
        V = (1-leak)*V + leak*np.tanh(np.dot(sample_input, W_in) + np.dot(V,W))
        V[0,bias] = 1.0
    facecolors[:] = cmap.to_rgba(V)
    collection.set_facecolors(facecolors)

animation = FuncAnimation(fig, update, interval=10)

#plt.tight_layout()
plt.show()
'''
