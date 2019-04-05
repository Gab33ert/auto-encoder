# topographique auto encoder sturcture
import os
import tqdm
import voronoi
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
from sklearn import preprocessing
np.set_printoptions(threshold=np.inf)


def connect(P, n_input_cells, n_output_cells, d, sigma):

    

    n = len(P)
    dP = P.reshape(1,n,2) - P.reshape(n,1,2)
    # Shifted Distances 
    D = np.hypot(dP[...,0]+d, dP[...,1])
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (np.random.uniform(0,1) < np.exp(-(D[j,i]**2)/(2*sigma**2)))& (P[i,0]<P[j,0]):
                W[j,i]=1  
    return W


def build(n_cells, n_input_cells = 32, n_output_cells = 32, sparsity = 0.01, seed=0):
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
    
    #np.random.seed(seed)
    density    = np.ones((1000,1000))
    n=1000
    for i in range(n):
	    ii=i/(n-1)
	    density[:,i]=np.power(((3.6)*ii*(ii-1)+1)*np.ones((1,n)),0.75) #neurone density
    density_P  = density.cumsum(axis=1)
    density_Q  = density_P.cumsum(axis=1)
    filename = "autoencoder-second-degree-big.npy"#"CVT-%d-seed-%d.npy" % (n_cells,seed)

    if not os.path.exists(filename):
        cells_pos = np.zeros((n_cells,2))
        cells_pos[:,0] = np.random.uniform(0, 1000, n_cells)
        cells_pos[:,1] = np.random.uniform(0, 1000, n_cells)
        for i in tqdm.trange(75):
            _, cells_pos = voronoi.centroids(cells_pos, density, density_P, density_Q)
        np.save(filename, cells_pos)

    cells_pos = np.load(filename)
    cells_in  = np.argpartition(cells_pos, +n_input_cells, 0)[:+n_input_cells]
    cells_out = np.argpartition(cells_pos, -n_output_cells, 0)[-n_output_cells:]
    
    W=connect(cells_pos, n_input_cells, n_output_cells, d, sigma)


    return cells_pos/1000, W, cells_in[:,0], cells_out[:,0]#, W_in, W_out, bias
    


def sigmoid(x):
    return 2/(1+np.exp(-x))-1

def dsigmoid(x):
    f=sigmoid(x)
    return -(f+1)*(f-1)/2

def forward(x, w, t):
    X=[]
    for i in range(t):
        X.append(x)
        x=sigmoid(w.dot(x))
    return X, x

def backward(x_in, w, t, in_index, out_index, alpha):
    X, x_out=forward(x_in, w, t)
    x_in_reshape=np.zeros(x_in.shape)
    x_in_reshape[out_index]=x_in[in_index]

    mask = np.ones(len(x_out), dtype=bool)
    mask[out_index] = False
    x_out_reshape=x_out
    x_out_reshape[mask]=0

    e=[dsigmoid(w.dot(X[t-1]))*(x_out_reshape-x_in_reshape)]
    for i in range(t-2,-1,-1):
        e.append(dsigmoid(w.dot(X[i]))*np.transpose(w).dot(e[t-2-i]))
    for i in range(t):
        for j in range(x_in.shape[1]):
            w-=alpha*np.outer(e[t-1-i][:,j], X[i][:,j])*W  #        looooooooooooooooooooook heeeeeeeeeeeeeeeeeere
    return  w, (np.sum(np.abs((x_out_reshape-x_in_reshape))))/(in_index.shape[0]*x_in.shape[1])

def err(x_in, w, t):
    X, x_out=forward(x_in, w, t)
    x_in_reshape=np.zeros(x_in.shape)
    x_in_reshape[out_index]=x_in[in_index]

    mask = np.ones(len(x_out), dtype=bool)
    mask[out_index] = False
    x_out_reshape=x_out
    x_out_reshape[mask]=0
    return (np.sum(np.abs(x_out_reshape-x_in_reshape)))/(in_index.shape[0]*x_in.shape[1]), np.max(x_out_reshape-x_in_reshape)

    
    
def train(x_in, w, t, in_index, out_index, iterr, alpha):
    error=np.zeros(iterr)
    c=0
    for i in range(iterr):
        #x_batch=x_in[:,c:c+32]
        #c+=32
        #if c>dataset_size-34:
        #    c=0
        alpha*=(10)**(1/(-iterr))
        w,  error[i] = backward(x_in, w, t, in_index, out_index, alpha)
        """
        if a==100:
            X, x=forward(x_in, w, t)
            plt.plot(np.linspace(-1, 1, size),scaled_data)
            plt.plot(np.linspace(-1, 1, size),x[out_index])
            plt.show()
            a=0
        a+=1
        """
    plt.semilogy(error)
    return w, error[iterr-1]

def generate_poly(data_size, n, degree):
    data=np.zeros((data_size, n))
    def poly(x, param):
        p=0
        for i in range(len(param)):
            p=p*x+param[i]
        return p
    for i in range(n):
        a=2*np.random.random([degree+1])-1
        data[:,i]=poly(np.linspace(-1, 1, data_size), a)
    return data


#global variale
size=70
n_cell=300
dataset_size=100
dataset_size_t=400
t=8
sigma=140
d=100
alpha=0.008
error=[]


P, W, in_index, out_index = build(n_cell, size, size,sparsity=0.05, seed=1)
wc=copy.deepcopy(W)
wc*=(2*np.random.random(wc.shape)-1)
    
    
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))#be careful the polynome are in [0,1] maybe you need [-1,1]
data=generate_poly(size, dataset_size, 40)
data_t=generate_poly(size, dataset_size_t, 40)
scaled_data=scaler.fit_transform(data)
scaled_data_t=scaler.fit_transform(data_t)
    
x=np.zeros((n_cell,dataset_size))
x[in_index]=scaled_data
x_t=np.zeros((n_cell,dataset_size_t))
x_t[in_index]=scaled_data_t
    
wc,e=train(x, wc, t, in_index, out_index, 500, alpha)


plt.show()
X, x=forward(x, wc, t)
plt.plot(np.linspace(-1, 1, size),scaled_data)
plt.plot(np.linspace(-1, 1, size),x[out_index])
plt.show()

X, x=forward(x_t[:,0:4], wc, t)


for i in range(3):
    plt.plot(np.linspace(-1, 1, size),scaled_data_t[:,i])
    plt.plot(np.linspace(-1, 1, size),x[:,i][out_index])
    plt.show()




print(e) 
print(err(x_t, wc, t)) 

"""


def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def forward(x, w):
    l=len(w)
    X=[]
    for i in range(l):
        X.append(x)
        x=sigmoid(w[i].dot(x))
    return X, x

def backprop(x_in, w, alpha): #propagate back, train W one step and resturn actual error
    X, x_out=forward(x_in,w)
    l=len(w)-1
    e=[dsigmoid(w[l].dot(X[l]))*(x_out-x_in)]
    for i in range(l-1,-1,-1):
        e.append(dsigmoid(w[i].dot(X[i]))*np.transpose(w[i+1]).dot(e[l-1-i]))
    for i in range(l+1):
        for j in range(x_in.shape[1]):
           w[i]-=alpha*np.outer(e[l-i][:, j], X[i][:, j])
    return w, (np.sum(np.abs((x_out-x_in))))/(x_in.shape[1]*x_in.shape[0])

def err(x_in,w):
    X, x_out=forward(x_in,w)
    return np.sum(np.abs((x_out-x_in)))/(x_in.shape[1]*x_in.shape[0])
    
def train(x_in, w, iterr, alpha,):
    error=np.zeros(iterr)
    for i in range(iterr):
            w, error[i] = backprop(x_in, w, alpha)
    plt.semilogy(error)
    return w, error[iterr-1]




def generate_poly(data_size, n, degree):
    data=np.zeros((data_size, n))
    def poly(x, param):
        p=0
        for i in range(len(param)):
            p=p*x+param[i]
        return p
    for i in range(n):
        a=2*np.random.random([degree+1])-1
        data[:,i]=poly(np.linspace(-2, 2, data_size), a)
    return data



# Build
# ------
#P, W, W_in, W_out, bias = build(1000, 32, 32, n_input=1, n_output=1,sparsity=0.05, seed=0)
 
l=5    
size=100

P, W = build(300, size, size, l=l, n_input=1, n_output=1,sparsity=0.05, seed=1)
w=copy.deepcopy(W)

w[0]=(2*np.random.random(w[0].shape)-1)*w[0]
for i in range(1,len(w)):
    w[i]=(2*np.random.random(w[i].shape)-1)*w[i]
    print(w[i].shape)


scaler = preprocessing.MinMaxScaler()
data=generate_poly(size, 40, 10)
data_t=generate_poly(size, 10, 10)
scaled_data=scaler.fit_transform(data)
scaled_data_t=scaler.fit_transform(data_t)
#unscaled_data=scaler.inverse_transform(scaled_data)

data=np.random.random([size,10])
a,b=train(scaled_data, w, 50000, 0.02)

X, x=forward(scaled_data,w)
print(b)
print("test", err(scaled_data_t,w))

#for i in range(6):
#    print(w[i]-W[i])
#print(w)

#print(b)
"""
"""
w=copy.deepcopy(W)

a,b=train(x, w, 10000, 3)
print(b)
for i in range(len(a)):
    print(a[i].shape)
"""
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