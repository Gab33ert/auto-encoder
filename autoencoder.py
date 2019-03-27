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
from random import randint
import copy

def connect(Pl, l, n_input_cells, n_output_cells):
    sigma=300
    W=[]
    q=1.2 #control the rate connection to first layer (has to be bigger than 1)
    win=np.floor(np.random.uniform(0,q,(len(Pl[0]),n_input_cells)))#create input connection
    for i in range(n_input_cells):#ensure each entry is connected to at least one neurone
        counter=0
        for j in range(len(Pl[0])):
            if win[j,i]!=0:
                counter+=1
        if counter==0:
            win[randint(0,len(Pl[0])-1),i]=1
    W.append(win)
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
    wout=np.floor(np.random.uniform(0,2,(n_output_cells,len(Pl[l-1]))))
    W.append(wout)
    
    return W

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def forward(x, w):
    l=len(w)
    X=[]
    for i in range(l):
        X.append(x)
        x=sigmoid(w[i].dot(np.transpose(x)))
    return X, x

def backprop(x_in, w, alpha): #propagate back, train W one step and resturn actual error
    X, x_out=forward(x_in,w)
    l=len(w)-1
    e=[dsigmoid(w[l].dot(X[l]))*(x_out-x_in)]
    for i in range(l-1,-1,-1):
        e.append(dsigmoid(w[i].dot(X[i]))*e[l-1-i].dot(w[i+1]))
    for i in range(l+1):
        w[i]-=W[i]*alpha*np.outer(e[l-i], X[i])
    return w, (np.sum(x_out-x_in))**2

    
def train(x_in, w, iterr, alpha):
    error=np.zeros(iterr)
    for i in range(iterr):
        w, error[i] = backprop(x_in, w, alpha)
    plt.loglog(error)
    return w, error[iterr-1]


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
	    density[:,i]=((3.5)*ii*(ii-1)+1)*np.ones((1,n)) #neurone density
    density_P  = density.cumsum(axis=1)
    density_Q  = density_P.cumsum(axis=1)
    filename = "autoencoder-second-degree.npy"#"CVT-%d-seed-%d.npy" % (n_cells,seed)

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

    
    W=connect(cells_pos, l, n_input_cells, n_output_cells)
    return cells_pos/1000, W#, W_in, W_out, bias
    

# Build
# ------
#P, W, W_in, W_out, bias = build(1000, 32, 32, n_input=1, n_output=1,sparsity=0.05, seed=0)
 
l=5    
size=10
P, W = build(40, size, size, l=l, n_input=1, n_output=1,sparsity=0.05, seed=0)
x=np.random.random([size])

w=copy.deepcopy(W)
a,b=train(x, w, 10000, 3)
for i in range(len(a)):
    print(a[i].shape)

"""
#print('{0:1.2e}'.format(b))
li=[0.5]
for i in range(3):
    li.append(li[i]*1.8)
c=[]
for i in li:
    w=copy.deepcopy(W)
    a,b=train(x, w, 10000, i)
    print(b, i)
    c.append(b)
plt.show()
plt.loglog(li, c)
"""