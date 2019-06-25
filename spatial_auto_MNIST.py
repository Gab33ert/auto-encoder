# This algorithm train spatial auto encoder on MNIST
# You can play with parameter sigma, d, wide...
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


#global variable
size=784
n_cell=3000
dataset_size=800
dataset_size_t=400
t=2
sigma=10#100
d=150
wide=0.01
alpha=0.0005
iterr=100


def connect(P, n_input_cells, n_output_cells, d, sigma):

    

    n = len(P)
    dP = P.reshape(1,n,2) - P.reshape(n,1,2)
    # Shifted Distances 
    D = np.hypot(dP[...,0]+d, wide*dP[...,1])
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (np.random.uniform(0,1) < np.exp(-(D[j,i]**2)/(2*sigma**2))) & (P[i,0]<P[j,0]):
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
    filename = "autoencoder-second-degree-MNIST.npy"#"CVT-%d-seed-%d.npy" % (n_cells,seed)

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
    out_index=visualize(W, cells_pos/1000, cells_in[:,0], t)

    return cells_pos/1000, W, cells_in[:,0], out_index#, W_in, W_out, bias
    


def sigmoid(x):
    return 2/(1+np.exp(-4*x))-1

def dsigmoid(x):
    f=sigmoid(x)
    return -4*(f+1)*(f-1)/2

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
    sparse_rate=[5]
    e=[dsigmoid(w.dot(X[t-1]))*(x_out_reshape-x_in_reshape)]
    for i in range(t-2,-1,-1):
        b=sparse_rate[i]*sigmoid(w.dot(X[i]))
        e.append(dsigmoid(w.dot(X[i]))*(np.transpose(w).dot(e[t-2-i])+b))
    for i in range(t):
        w-=alpha*(e[t-1-i].dot(np.transpose(X[i])))*W
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

    
    
def train(x_in, x_t, w, t, in_index, out_index, iterr, alpha):
    error=np.zeros(iterr)
    #c=0
    for i in range(iterr):
        print(i)
        #x_batch=x_in[:,c:c+32]
        #c+=32
        #if c>dataset_size-34:
        #    c=0
        alpha*=(5)**(1/(-iterr))
        w,  error[i] = backward(x_in, w, t, in_index, out_index, alpha)
        error[i]=err(x_t,w,t)[0]
    return w, error

def visualize(W, P, in_index, t):
    index=in_index
    plt.scatter(P[index][:,0],P[index][:,1])
    axes = plt.gca()
    axes.set_xlim([0,1])
    plt.show()      
    for i in range(t):
        x=np.zeros(W.shape[0])
        for j in index:x[j]=1
        x=W.dot(x)
        x=(x > 0).astype(int)
        index=[idx for idx, v in enumerate(x) if v]
        plt.scatter(P[index][:,0],P[index][:,1])
        axes = plt.gca()
        axes.set_xlim([0,1])
        plt.show()
    index=np.array(index)[np.argsort(P[index][:,0])][len(index)-in_index.shape[0]:len(index)]
    plt.scatter(P[index][:,0],P[index][:,1], s=1)
    axes = plt.gca()
    axes.set_xlim([0,1])
    plt.show()
    return index
 
def animation(n, t, X, x, P):
    XX=[]
    for i in range(t):
        XX.append(X[i][:,n])
    XX.append(x[:,n])
    f=plt.figure()
    plt.scatter(P[out_index][:,0],P[out_index][:,1], color="black", s=7)
    l= plt.scatter(P[:,0],P[:,1], c=X[0][:,n], cmap='PiYG', vmin=-1, vmax=1, s=5)
    
    def update(i):
        l.set_array(XX[i])
        return l,
    ami=FuncAnimation(f, update, frames=t+1, interval=400, blit=True, repeat=True)
    ami.save("lala.gif", writer="imagemagick")
    plt.show()
    
    for i in range(t):
        plt.scatter(P[out_index][:,0],P[out_index][:,1], color="black", s=7)
        plt.scatter(P[:,0],P[:,1], c=X[i][:,n], cmap='PiYG', vmin=-1, vmax=1, s=5)
        plt.colorbar()
        plt.show()
    plt.scatter(P[out_index][:,0],P[out_index][:,1], color="black", s=7)
    plt.scatter(P[:,0],P[:,1], c=x[:,n], cmap='PiYG', vmin=-1, vmax=1, s=5)
    plt.colorbar()
    plt.show()    

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



error=[]
P, W, in_index, out_index = build(n_cell, size, size,sparsity=0.05, seed=1)  #generate topology and network
wc=copy.deepcopy(W)
wc*=(2*np.random.random(wc.shape)-1)
#wc=np.load("test.npy")
#W=copy.deepcopy((wc != 0 ).astype(int))

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train=x_train[0:dataset_size,:,:]
x_test=x_test[0:dataset_size_t,:,:]
x_train, x_test = (x_train / 127.5) - 1, (x_test / 127.5) - 1#x_train, x_test = (x_train / 255), (x_test / 255)
    
x_t=np.zeros((n_cell,dataset_size_t))
x_test=np.transpose(np.asarray(x_test).reshape(400,-1))
x_t[in_index]=x_test
X, x=forward(x_t, wc, t)

for i in range(2):                                                      #plot mean neuron activation, layer size and mean number of activated neuron(product of the 2 former) in each layer
    a=(np.mean(np.abs(X[i]), axis=1)[np.argwhere(np.mean(np.abs(X[i]), axis=1)!=0)])
    print(np.mean(a), a.shape[0], a.shape[0]*np.mean(a))
a=(np.mean(np.abs(x), axis=1)[np.argwhere(np.mean(np.abs(x), axis=1)!=0)])
print(np.mean(a), a.shape[0], a.shape[0]*np.mean(a))


  
x=np.zeros((n_cell,dataset_size))
x_train=np.transpose(np.asarray(x_train).reshape(dataset_size,-1))
x[in_index]=x_train


wc,e=train(x, x_t, wc, t, in_index, out_index, iterr, alpha)                #train the network

fig = plt.figure()                                                          #plot images and their reconstructions
for i in range(10):
    if i<5:fig.add_subplot(4,5,1+i)
    else:fig.add_subplot(4,5,6+i)
    plt.axis('off')
    plt.imshow(x_test[:,i].reshape(28,28))
    im=np.zeros(n_cell)
    im[in_index]=x_test[:,i]
    IM, im=forward(im, wc, t)
    if i<5:fig.add_subplot(4,5,6+i)
    else:fig.add_subplot(4,5,11+i)
    plt.axis('off')
    plt.imshow(im[out_index.reshape(28,28)])
#plt.savefig("mnistAuto1.pdf")
plt.show()

print(err(x, wc, t)) 
print(err(x_t, wc, t)) 
x=np.zeros((n_cell,x_test.shape[1]))
x[in_index]=x_test
X, x=forward(x, wc, t)

for i in range(2):                                                      #plot mean neuron activation, layer size and mean number of activated neuron(product of the 2 former) in each layer
    a=(np.mean(np.abs(X[i]), axis=1)[np.argwhere(np.mean(np.abs(X[i]), axis=1)!=0)])
    print(np.mean(a), a.shape[0], a.shape[0]*np.mean(a))
a=(np.mean(np.abs(x), axis=1)[np.argwhere(np.mean(np.abs(x), axis=1)!=0)])
print(np.mean(a), a.shape[0], a.shape[0]*np.mean(a))

fig=plt.figure()                                                        #plot learning curve
ax=fig.add_subplot(111)
plt.plot(e)
ax.set_ylabel('Mean absolute value error')
ax.set_xlabel('Iterration')
ax.set_ylim(0, 0.65)
#plt.savefig("autoMnistCurve.pdf")
print(e[len(e)-10: len(e)-1])
