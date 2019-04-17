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
import tensorflow as tf


#build graph
def connect(P, n_input_cells, n_output_cells, d, sigma):

    

    n = len(P)
    dP = P.reshape(1,n,2) - P.reshape(n,1,2)
    # Shifted Distances 
    D = np.hypot(dP[...,0]+d, dP[...,1])
    #W = np.zeros((n,n))
    W=np.where((np.random.uniform(0,1,(n,n)) < np.exp(-(D**2)/(2*sigma**2))), 1, 0)
    s=np.argwhere(W==1)
    for i in range(s.shape[0]):
        if(P[s[i,1],0]>P[s[i,0],0]):
            W[s[i,0],s[i,1]]=0
    """
    for i in range(n):
        for j in range(n):
            if (np.random.uniform(0,1) < np.exp(-(D[j,i]**2)/(2*sigma**2))) & (P[i,0]<P[j,0]):
                W[j,i]=1
    """
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
	    density[:,i]=np.power((ii*(ii-2)+1)*np.ones((1,n)),6) #neurone density
    density_P  = density.cumsum(axis=1)
    density_Q  = density_P.cumsum(axis=1)
    filename = "autoencoder-rbm-MNIST.npy"#"CVT-%d-seed-%d.npy" % (n_cells,seed)
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

    return cells_pos/1000, W, cells_in[:,0], cells_out[:,0]
    
def abstract_layer(in_index, W, t):#unfold the total connection matrix into layer by layer connection matrix
    index=in_index
    Wt=[]
    for i in range(t-1):
        x=np.zeros(W.shape[0])
        for i in index:x[i]=1
        x=W.dot(x)
        x=(x > 0).astype(int)
        index_new=np.asarray([idx for idx, v in enumerate(x) if v])
        Wt.append(W[index_new[:, None], index])
        index=index_new
    return Wt


#BACKPROP
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
        w-=alpha*e[t-1-i].dot(np.transpose(X[i]))*W
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

    
    
def train_backprop(x_in, w, t, in_index, out_index, iterr, alpha):
    error=np.zeros(iterr)
    c=0
    for i in range(iterr):
        print(i)
        #x_batch=x_in[:,c:c+32]
        #c+=32
        #if c>dataset_size-34:
        #    c=0
        alpha*=(5)**(1/(-iterr))
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
    plt.show()
    return w, error[iterr-1]


#RBM
def sample_rbm_forward(visible, c, w):
    return np.where(np.random.rand(w.shape[0],visible.shape[1]) < sigmoid(np.tile(c,(1,visible.shape[1]))+w.dot(visible)), 1, 0)

def sample_rbm_backward(hidden, b, w):
    return np.random.normal(np.tile(b,(1,hidden.shape[1]))+np.transpose(w).dot(hidden),1)#np.where(np.random.rand(w.shape[1],hidden.shape[1]) < sigmoid(np.tile(b,(1,hidden.shape[1]))+np.transpose(w).dot(hidden)), 0, 1) #this is no more sampling!!!!!!!!!!!!!

def backandforw(hidden, b, c, w, k):
    for i in range(k):
        visible_k=sample_rbm_backward(hidden, b, w)
        hidden=sample_rbm_forward(visible_k, c, w)
        return hidden, visible_k

def actualise_weight(visible, b, c, w, n, k):
    hidden=sample_rbm_forward(visible, c, w)
    hidden_c, visible_c=backandforw(hidden, b, c, w, k)
    #if n%10==0:print(np.mean(hidden,1))
    #print((hidden.dot(np.transpose(visible)))[1,:]-(hidden.dot(np.transpose(visible)))[2,:])
    #f=np.zeros((hidden.shape[0], visible.shape[0]))
    #g=np.zeros((hidden.shape[0], 1))
    #f[(n)%10,:]=np.ones((1,visible.shape[0]))
    #g[n%10]=1
    w+=(1/(visible.shape[1]))*epsilon_w*(hidden.dot(np.transpose(visible))-hidden_c.dot(np.transpose(visible_c)))#*f
    b+=(1/(visible.shape[1]))*epsilon_b*(np.sum(visible-visible_c, axis=1)).reshape(visible.shape[0],1)
    c+=(1/(visible.shape[1]))*epsilon_c*(np.sum(hidden-hidden_c, axis=1)).reshape(w.shape[0],1)#*g
    e=0
    for i in range(visible.shape[1]):
        e-=(hidden[:,i]).dot(w.dot(visible[:,i]))
    e+=-np.sum(b*visible)-np.sum(c*hidden)
    return e/(visible.shape[1]), np.sum(np.sum(np.abs(visible-visible_c)))/(visible.shape[0]*visible.shape[1])
  
def train_rbm(visible, b, c, w, iterr_rbm):
    e=[]
    E=[]
    d=0
    f=0
    for i in range(iterr_rbm):
        visible_batch=visible[:,d:d+32]
        #visible_batch=visible_batch.reshape(visible.shape[0],1)
        f+=1
        if f==32:
            d+=32
            f=0
        if d>dataset_size-34:
            d=0
        a, b = actualise_weight(visible_batch, b, c, w, i+1, 1)
        e.append(a)
        E.append(b)
    """
    for i in range(iterr_rbm//3):
        a, b = actualise_weight(visible, b, c, w, i+1, 3)
        e.append(a)
        E.append(b)
    """
    plt.plot(e)
    plt.yscale('symlog')
    plt.show()
    plt.plot(E)
    plt.yscale('symlog')
    plt.show()
    
def gibbs_sampling(visible, b, c, w, n):
    a=visible
    plt.imshow(visible.reshape(28,28))
    plt.show()
    for i in range(n):
        hidden=sample_rbm_forward(a, c, w)
        visible=sample_rbm_backward(hidden, b, w)
    plt.imshow(visible.reshape(28,28))
    plt.show()

def error(visible, b,c,w):
    return np.sum(np.abs(visible-sample_rbm_backward(sample_rbm_forward(visible,c,w),b,w)))/(visible.shape[0]*visible.shape[1])



#TOOLS
def visualize(in_index, t):
    index=in_index
    l=[]
    for i in range(t):
        l.append(len(index))
        plt.scatter(P[:,0],P[:,1], s=3)
        plt.scatter(P[index][:,0],P[index][:,1], s=3)
        axes = plt.gca()
        axes.set_xlim([0,1])
        plt.show()
        x=np.zeros(W.shape[0])
        for i in index:x[i]=1
        x=W.dot(x)
        x=(x > 0).astype(int)
        index=[idx for idx, v in enumerate(x) if v]
    print(l)
 


#global variable
size=784
n_cell=1600
dataset_size=800
dataset_size_t=400
t=5
sigma=50
d=310
alpha=0.0005#backprop rate

epsilon_w=0.01#rbm rate
epsilon_b=epsilon_w
epsilon_c=epsilon_w
iterr_rbm=100


P, W, in_index, out_index = build(n_cell, size, size,sparsity=0.05, seed=1)

Wt=abstract_layer(in_index, W, t)
wc=copy.deepcopy(Wt[0])

wc=wc*(2*np.random.random(wc.shape)-1)*0.01
c=0.01*np.random.randn(wc.shape[0],1)
b=0.01*np.random.randn(wc.shape[1],1)

#wc=np.load("test.npy")
#W=copy.deepcopy((wc != 0 ).astype(int))

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train=x_train[0:dataset_size,:,:]
x_test=x_test[0:dataset_size_t,:,:]


x_train=np.transpose(np.asarray(x_train).reshape(dataset_size,-1))
x_train=preprocessing.scale(x_train, axis=1)

    
"""
x=np.zeros((n_cell,dataset_size))
x_train=np.transpose(np.asarray(x_train).reshape(dataset_size,-1))
x[in_index]=x_train
x_t=np.zeros((n_cell,dataset_size_t))
x_test=np.transpose(np.asarray(x_test).reshape(400,-1))
x_t[in_index]=x_test
"""

train_rbm(x_train, b, c, wc, iterr_rbm)
for i in range(5):
    plt.imshow(wc[i,:].reshape(28,28))
    plt.show()
    #plt.imshow(x_train[:,i].reshape(28,28))
    #plt.show()
print(error(x_train,b,c,wc))
gibbs_sampling(x_train[:,0:1], b, c, wc, 10)

"""
wc,e=train_backprop(x, wc, t, in_index, out_index, 100, alpha)

for i in range(5):
    plt.imshow(x_test[:,i].reshape(28,28))
    plt.show()
    im=np.zeros(n_cell)
    im[in_index]=x_test[:,i]
    IM, im=forward(im, wc, t)
    plt.imshow(im[out_index.reshape(28,28)])
    plt.show()

print(err(x, wc, t)) 
print(err(x_t, wc, t)) 
"""

"""
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





"""
