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
import time
from numpy.core.umath_tests import inner1d
from mpl_toolkits.mplot3d import Axes3D as ax3

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

def connect_3d(P, d, sigma):
    n = len(P)
    dP = P.reshape(1,n,3) - P.reshape(n,1,3)
    # Shifted Distances 
    D = np.hypot(dP[...,0], dP[...,1])
    D = np.hypot(D, dP[...,2]+d)
    #W = np.zeros((n,n))
    W=np.where((np.random.uniform(0,1,(n,n)) < np.exp(-(D**2)/(2*sigma**2))), 1, 0)
    s=np.argwhere(W==1)
    for i in range(s.shape[0]):
        if(P[s[i,1],2]>=P[s[i,0],2]):
            W[s[i,0],s[i,1]]=0
    return W

def build_3d(n):
    i=28
    cells_pos=np.zeros((n*i*i,3))
    for z in range(n):
        for y in range(i):
            for x in range(i):
                if np.random.random()<((n-z)/n)**2:
                    cells_pos[x+i*(y+i*z),0]=x*1000/(i-1)
                    cells_pos[x+i*(y+i*z),1]=y*1000/(i-1)
                    cells_pos[x+i*(y+i*z),2]=z*1000/(i-1)
    cells_pos=np.concatenate((np.zeros((1,3)),cells_pos[np.argwhere(cells_pos[:,0]+cells_pos[:,1]+cells_pos[:,2]),:][:,0,:]))
    cells_pos[:,2]+=(1000/27)*np.random.random(cells_pos[:,2].shape)
    return cells_pos/1000, connect_3d(cells_pos, d, sigma), np.arange(28*28)


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
    return (2/(1+np.exp(-x)))-1

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
    return np.where(np.random.rand(w.shape[0],visible.shape[1]) < sigmoid(np.tile(c,(1,visible.shape[1]))+w.dot(visible)), 1, -1)

def sample_rbm_backward(hidden, c, w):
    return np.where(np.random.rand(w.shape[1],hidden.shape[1]) < sigmoid(np.tile(c,(1,hidden.shape[1]))+np.transpose(w).dot(hidden)), 1, -1)

def sample_grbm_backward(hidden, b, w):
    return np.random.normal(np.tile(b,(1,hidden.shape[1]))+np.transpose(w).dot(hidden),0.01)#np.where(np.random.rand(w.shape[1],hidden.shape[1]) < sigmoid(np.tile(b,(1,hidden.shape[1]))+np.transpose(w).dot(hidden)), 0, 1) #this is no more sampling!!!!!!!!!!!!!

def backandforw(hidden, b, c, w, k):
    for i in range(k):
        if mode==0:
            visible_k=sample_grbm_backward(hidden, b, w)
        if mode==1:
            visible_k=sample_rbm_backward(hidden, b, w)
        hidden=sample_rbm_forward(visible_k, c, w)
    return hidden, visible_k

def actualise_weight(visible, b, c, w, n, k, t):
    hidden=sample_rbm_forward(visible, c, w)
    hidden_c, visible_c=backandforw(hidden, b, c, w, k)
    w+=(1/(visible.shape[1]))*epsilon_w*(hidden.dot(np.transpose(visible))-hidden_c.dot(np.transpose(visible_c)))#*Wt[t]
    b+=(1/(visible.shape[1]))*epsilon_b*(np.sum(visible-visible_c, axis=1)).reshape(visible.shape[0],1)
    c+=(1/(visible.shape[1]))*epsilon_c*(np.sum(hidden-hidden_c, axis=1)).reshape(w.shape[0],1)#*g

    
def train_rbm(visible, b, c, w, iterr_rbm, t):
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
        actualise_weight(visible_batch, b, c, w, i+1, 1, t)
        if i%50==0:
            e.append(energy(x_test, b, c, w))
            E.append(energy(visible, b, c, w))
    """
    for i in range(iterr_rbm//5):
        actualise_weight(visible, b, c, w, i+1, 5, t)
        if i%10==0:
            print(i)
            e.append(energy(x_test, b, c, w))
            E.append(energy(x_train, b, c, w))    
    """
    
    plt.plot(e,label="test set")
    plt.plot(E, label="training set")
    plt.legend(loc='upper right')
    plt.yscale('symlog')
    plt.show()
    print("energy", e[len(e)-1])


def energy(visible, b, c, w):
    e=0
    hidden=sample_rbm_forward(visible, c, w)
    e+=-np.sum(b*visible)-np.sum(c*hidden)-np.sum(inner1d(hidden, w.dot(visible)))
    return e/(visible.shape[1]) 
        
 
def gibbs_sampling(visible, b, c, w, n):
    plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(28,28), vmin=-100, vmax= 400)
    plt.colorbar()
    plt.show()
    a=visible
    for i in range(n):
        hidden=sample_rbm_forward(a, c, w)
        visible=sample_grbm_backward(hidden, b, w)
    plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(28,28), vmin=-100, vmax= 400)
    plt.colorbar()
    plt.show()
    
def gibbs_deep_sampling(visible, b, c, w, n):
    #plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(28,28), vmin=-100, vmax= 400)
    #plt.colorbar()
    #plt.show()
    a=visible
    hidden=sample_rbm_forward(a, c[0], w[0])
    for i in range(n):
        hidden=sample_rbm_forward(hidden, c[i+1], w[i+1])
    for i in range(n):
        hidden=sample_rbm_backward(hidden, b[n-i], w[n-i])
    visible=sample_grbm_backward(hidden, b[0], w[0])
    plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(28,28), vmin=-100, vmax= 400)
    plt.colorbar()
    plt.show()
      
def fake_data(b, c, w, n):
    hidden=np.random.randint(2, size=c[n].shape)
    hidden, visible=backandforw(hidden, b[n], c[n], w[n], 50)
    N=n
    while N >1:
        visible=sample_rbm_backward(visible, b[N-1], w[N-1])
        N-=1
    plt.imshow(sample_grbm_backward(visible, b[N-1], w[N-1]).reshape(28,28))
    

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
 
def visualize_3d(in_index, t):
    index=in_index
    l=[]
    for i in range(t):
        l.append(len(index))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(P[index][:,0],P[index][:,1], P[index][:,2])
        ax.scatter(P[:,0],P[:,1], P[:,2], s=1)
        plt.show()
        x=np.zeros(W.shape[0])
        for i in index:x[i]=1
        x=W.dot(x)
        x=(x > 0).astype(int)
        index=[idx for idx, v in enumerate(x) if v]
    print(l)

    
def analyze_topology_back(Wt, depth):
    n=Wt[depth-1].shape[0]
    l=[]
    for j in range(n):
        out=np.zeros((n,1))
        out[j,0]=1
        for i in range(depth):
            out=np.transpose(out).dot(Wt[depth-1-i])
            out=np.transpose((out > 0).astype(int))
        l.append(100*np.sum(out)/out.shape[0])
    plt.hist(l)
    plt.title("backward depth "+str(depth))
    #plt.ylim(bottom=0)
    plt.show()
    
def analyze_topology(Wt, depth):
    n=Wt[depth-1].shape[0]
    out=np.ones((n,1))
    for i in range(depth):
        out=np.transpose(out).dot(Wt[depth-1-i])
        out=np.transpose((out > 0).astype(int))
    print(100*np.sum(out)/out.shape[0])
    

def analyze_topology_froward(Wt, depth):
    l=[]    
    n=Wt[0].shape[1]
    for j in range(n):
        x=np.zeros((n,1))
        x[j,0]=1
        for i in range(depth):
            x=Wt[i].dot(x)
            x=(x > 0).astype(int)
        l.append(100*np.sum(x)/x.shape[0])
    plt.hist(l)
    plt.title("forward depth "+str(depth))
    #plt.ylim(bottom=0)
    plt.show()

 

#global variable
size=784
n_cell=1600
dataset_size=1000
dataset_size_t=300
t=5
sigma=90
d=260
alpha=0.0005#backprop rate

epsilon_w=0.001#rbm rate
epsilon_b=epsilon_w
epsilon_c=epsilon_w


P, W,in_index =build_3d(5)
Wt=abstract_layer(in_index, W, t)
"""
analyze_topology(Wt, 4)
analyze_topology(Wt, 3)
analyze_topology(Wt, 2)
analyze_topology(Wt, 1)

analyze_topology_froward(Wt,4)
analyze_topology_froward(Wt,3)
analyze_topology_froward(Wt,2)
analyze_topology_froward(Wt,1)

analyze_topology_back(Wt,4)
analyze_topology_back(Wt,3)
analyze_topology_back(Wt,2)
analyze_topology_back(Wt,1)
"""

f, axarr = plt.subplots(4, sharex=True)                                        #histogram of the number of output neurone recieving 1,2... input neurone at each layer
axarr[0].hist(np.sum(Wt[0],axis=1), bins=np.max(np.sum(Wt[0],axis=1)))
axarr[1].hist(np.sum(Wt[1],axis=1), bins=np.max(np.sum(Wt[1],axis=1)))
axarr[2].hist(np.sum(Wt[2],axis=1), bins=np.max(np.sum(Wt[2],axis=1)))
axarr[3].hist(np.sum(Wt[3],axis=1), bins=np.max(np.sum(Wt[3],axis=1)))
plt.show()

w=[]                                                                           #initialise weight and bias
b=[]#forward bias
c=[]#backward bias
for i in range(len(Wt)):
    wc=copy.deepcopy(Wt[i])
    wc=np.ones(wc.shape)
    wc=wc*np.random.normal(0, 0.01, wc.shape)#(2*np.random.random(wc.shape)-1)*0.01
    w.append(wc)
    b.append(np.zeros((wc.shape[1],1)))
    c.append(np.zeros((wc.shape[0],1)))
b[0]=0.01*np.random.randn(Wt[0].shape[1],1)



#wc=np.load("test.npy")
#W=copy.deepcopy((wc != 0 ).astype(int))

mnist = tf.keras.datasets.mnist                                                #data set loading and scalling
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train=x_train[0:dataset_size,:,:]
x_test=x_test[0:dataset_size_t,:,:]
x_train=np.asarray(x_train).reshape(dataset_size,-1)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = np.transpose(scaler.transform(x_train))
#x_train=preprocessing.scale(x_train, axis=1)
x_test=np.asarray(x_test).reshape(dataset_size_t,-1)
x_test=np.transpose(scaler.transform(x_test))#np.transpose(preprocessing.scale(x_test, axis=1))


x_test_copy=x_test

visualize_3d(in_index, t)

answer = input("Do you want to keep going y/n?")
if answer == "y":
    mode=0#grbm mode
    #layer 1
    seconds = time.time()
    iterr_rbm=2000
    epsilon_w=0.001#rbm rate
    epsilon_b=epsilon_w
    epsilon_c=epsilon_w
    train_rbm(x_train, b[0], c[0], w[0], iterr_rbm, 0)
    for i in range(w[0].shape[0]-10, w[0].shape[0]-5):
        plt.imshow(w[0][i,:].reshape(28,28))
        plt.colorbar()
        plt.show()
    print("error", error(x_test,b[0],c[0],w[0]))
    print("time",time.time()-seconds)
    gibbs_sampling(x_test[:,0:1], b[0],c[0],w[0], 2)
    gibbs_sampling(x_test[:,1:2], b[0],c[0],w[0], 2)
    gibbs_sampling(x_test[:,20:21], b[0],c[0],w[0], 2)
    
    mode=1#rbm mode
    
    #layer 2
    seconds = time.time()
    iterr_rbm=1000
    epsilon_w=0.01#rbm rate
    epsilon_b=epsilon_w
    epsilon_c=epsilon_w
    x_train_1=sample_rbm_forward(x_train, c[0], w[0])
    x_test=sample_rbm_forward(x_test, c[0], w[0])
    train_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, 1)
    print("layer 2")
    print("error", error(x_test,b[1],c[1],w[1]))
    print("time",time.time()-seconds)

    #layer 3
    seconds = time.time()
    iterr_rbm=1000
    epsilon_w=0.005#rbm rate
    epsilon_b=epsilon_w
    epsilon_c=epsilon_w
    x_train_2=sample_rbm_forward(x_train_1, c[1], w[1])
    x_test=sample_rbm_forward(x_test, c[1], w[1])
    train_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm, 2)
    print("layer 3")
    print("error", error(x_test,b[2],b[3],w[2]))
    print("time",time.time()-seconds)

    #layer 4
    seconds = time.time()
    iterr_rbm=5000
    epsilon_w=0.001#rbm rate
    epsilon_b=epsilon_w
    epsilon_c=epsilon_w
    x_train_3=sample_rbm_forward(x_train_2, c[2], w[2])
    x_test=sample_rbm_forward(x_test, c[2], w[2])
    train_rbm(x_train_3, b[3],c[3],w[3], iterr_rbm, 3)
    print("layer 4")
    print("error", error(x_test,b[3],c[3],w[3]))
    print("time",time.time()-seconds)
    
    
    gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 0)
    gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 1)
    gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 2)
    gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 3)
    
elif answer == "n":
    print("ok")
else:
    print("Please enter y or n.")


