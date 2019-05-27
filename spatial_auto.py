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


def connect(P, n_input_cells, n_output_cells, d, sigma, wide):
    c=0
    b=0
    n = len(P)
    dP = P.reshape(1,n,2) - P.reshape(n,1,2)
    # Shifted Distances 
    D = np.hypot(dP[...,0]+d, wide*dP[...,1])
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (np.random.uniform(0,1) < np.exp(-(D[j,i]**2)/(2*sigma**2))):# & (P[i,0]<P[j,0]):
                W[j,i]=1 
                b+=1
                if (P[i,0]>P[j,0]):
                    c+=1
    print("c",c, "b", b)
    return W


def build(n_cells, n_input_cells = 32, n_output_cells = 32, wide=0.05, sparsity = 0.01, seed=0):
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
    #cells_out = np.argpartition(cells_pos, -n_output_cells, 0)[-n_output_cells:]
    in_index=cells_in[:,0]
    in_index=in_index[np.argsort(cells_pos[in_index][:,1])]
    
    W=connect(cells_pos, n_input_cells, n_output_cells, d, sigma, wide)
    out_index=visualize(W, cells_pos, cells_in[:,0], t)

    return cells_pos/1000, W, in_index, out_index#cells_out[:,0]#, W_in, W_out, bias
    
def unspatial(W, in_index):#completly rewire each layer
    index=in_index
    for i in range(t-1):
        x=np.zeros(W.shape[0])
        for j in index:x[j]=1
        x=W.dot(x)
        x=(x > 0).astype(int)
        index_new=np.asarray([idx for idx, v in enumerate(x) if v])
        W[index_new[:, None], index]=np.ones( W[index_new[:, None], index].shape)
        index=index_new
    return(W)


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

def backward(x_in, w, t, in_index, out_index, alpha):#for i in range(7):print(np.mean(np.mean(np.abs(X[i]), axis=1)[np.argwhere(np.mean(np.abs(X[i]), axis=1)!=0)]))
    X, x_out=forward(x_in, w, t)
    ro=[]
    idxx=[]
    for i in X:
        ii=np.mean(i, axis=1)
        idx=np.argwhere(ii!=0)
        idxx.append(idx)
        ro.append(ii[idx])
    
    beta=[0, 0, 0, 0, 0, 0]
    x_in_reshape=np.zeros(x_in.shape)
    x_in_reshape[out_index]=x_in[in_index]

    mask = np.ones(len(x_out), dtype=bool)
    mask[out_index] = False
    x_out_reshape=x_out
    x_out_reshape[mask]=0

    e=[dsigmoid(w.dot(X[t-1]))*(x_out_reshape-x_in_reshape)]
    for i in range(t-2,-1,-1):
        b=beta[i]*sigmoid(w.dot(X[i]))
        e.append(dsigmoid(w.dot(X[i]))*(np.transpose(w).dot(e[t-2-i])+b))#+np.transpose(b))
    for i in range(t):
        w-=alpha*((e[t-1-i]).dot(np.transpose(X[i]))+w)*W
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
    for i in range(iterr):
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
    return w, error[iterr-1]

def visualize(W, P, in_index, t):
    index=in_index
    plt.scatter(P[index][:,0],P[index][:,1])
    axes = plt.gca()
    axes.set_xlim([0,1000])
    plt.show()
    l=[]
    for i in range(t-1):
        x=np.zeros(W.shape[0])
        for j in index:x[j]=1
        print(np.sum(x))
        x=W.dot(x)
        x=(x > 0).astype(int)
        index_n=[idx for idx, v in enumerate(x) if v]
        l.append([np.sum(W[index_n,:][:,index]),len(index)*len(index_n)])
        index=index_n
        plt.scatter(P[index][:,0],P[index][:,1])
        axes = plt.gca()
        axes.set_xlim([0,1000])
        plt.show()
    index=np.array(index)[np.argsort(P[index][:,0])][len(index)-in_index.shape[0]:len(index)]
    index=index[np.argsort(P[index][:,1])]
    print(l)
    return index
    
def abstract_layer(in_index, W, t):                                            #unfold the total connection matrix into layer by layer connection matrix
    index=in_index
    Wt=[]
    for i in range(t-1):
        x=np.zeros(W.shape[0])
        for j in index:x[j]=1
        x=W.dot(x)
        x=(x > 0).astype(int)
        index_new=np.asarray([idx for idx, v in enumerate(x) if v])
        Wt.append(W[index_new[:, None], index])
        index=index_new
        print(Wt[i].shape)
    return Wt

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


#global variable
size=30
n_cell=300
dataset_size=500
dataset_size_t=400
t=7
sigma=15
d=100
alpha=0.001
error=[]
sparsity=[]
connection_use=[]
iterr=1000
#wide=[0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
wide=0.05
dd=[100]#, 50, 25, 12, 6, 3]
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))#be careful the polynome are in [0,1] maybe you need [-1,1]
data=generate_poly(size, dataset_size, 4)
data_t=generate_poly(size, dataset_size_t, 4)
scaled_data=scaler.fit_transform(data)
scaled_data_t=scaler.fit_transform(data_t)
    




for d in dd:
    P, W, in_index, out_index = build(n_cell, size, size,  wide, sparsity=0.05, seed=1)
    #unspatial(W, in_index)
    x=np.zeros((n_cell,dataset_size))
    x[in_index]=scaled_data
    x_t=np.zeros((n_cell,dataset_size_t))
    x_t[in_index]=scaled_data_t
    
    wc=copy.deepcopy(W)
    wc*=(2*np.random.random(wc.shape)-1)
    ws=copy.deepcopy(wc)
        
    
       
    wc,e=train(x, wc, t, in_index, out_index, iterr, alpha)#iter 1000
    plt.show()
    
    X, x=forward(x_t[:,0:4], wc, t)
    for i in range(3):
        plt.plot(np.linspace(-1, 1, size),scaled_data_t[:,i])
        plt.plot(np.linspace(-1, 1, size),x[:,i][out_index])
        plt.show()
    X, x=forward(x_t, wc, t)
    
    
    print(e) 
    print(err(x_t, wc, t)) 
    error.append(err(x_t, wc, t)[0])
    amin=100
    for i in range(7):
        a=np.mean(np.abs(X[i]), axis=1)[np.argwhere(np.mean(np.abs(X[i]), axis=1)!=0)]
        print(np.mean(a), a.shape[0], a.shape[0]*np.mean(a))
        if i==0:amax=a.shape[0]*np.mean(a)
        if a.shape[0]*np.mean(a)<amin:amin=a.shape[0]*np.mean(a)
    sparsity.append(amin/amax)
        
    p=np.sum(np.where(ws-wc!=0, 1, 0))
    pp=np.sum(np.where(ws!=0,1,0))
    print(p, pp, p/pp)
    connection_use.append(p/pp)
"""
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylabel('error')
ax1.set_xlabel('d')
plt.plot(dd, error)
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylabel('sparsity')
ax1.set_xlabel('d')
plt.plot(dd, sparsity)
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylabel('error')
ax1.set_xlabel('sparsity')
plt.plot(sparsity, error)
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylabel('connection_use')
ax1.set_xlabel('d')
plt.plot(dd, connection_use)
plt.show()
"""
"""
visualize(in_index[10:11],15)
"""    
"""fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylabel('error')
ax1.set_xlabel('wide')
plt.plot(wide, error)
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylabel('sparsity')
ax1.set_xlabel('wide')
plt.plot(wide, sparsity)
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylabel('error')
ax1.set_xlabel('sparsity')
plt.plot(sparsity, error)
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_ylabel('connection_use')
ax1.set_xlabel('wide')
plt.plot(wide, connection_use)
plt.show()"""