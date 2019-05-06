#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:56:41 2019

@author: gouraud
"""
import matplotlib.pyplot as plt
import numpy as np
import time

import function as func
import rbm_visualize as rbmv


#RBM
def sample_rbm_forward(visible, c, w):
    return np.where(np.random.rand(w.shape[0],visible.shape[1]) < func.sigmoid(np.tile(c,(1,visible.shape[1]))+w.dot(visible)), 1, -1)

def sample_rbm_backward(hidden, c, w):
    return np.where(np.random.rand(w.shape[1],hidden.shape[1]) < func.sigmoid(np.tile(c,(1,hidden.shape[1]))+np.transpose(w).dot(hidden)), 1, -1)

def sample_grbm_backward(hidden, b, w):
    return np.random.normal(np.tile(b,(1,hidden.shape[1]))+np.transpose(w).dot(hidden),0.01)#np.where(np.random.rand(w.shape[1],hidden.shape[1]) < sigmoid(np.tile(b,(1,hidden.shape[1]))+np.transpose(w).dot(hidden)), 0, 1) #this is no more sampling!!!!!!!!!!!!!

def backandforw(hidden, b, c, w, k, mode):
    for i in range(k):
        if mode==0:
            visible_k=sample_grbm_backward(hidden, b, w)
        if mode==1:
            visible_k=sample_rbm_backward(hidden, b, w)
        hidden=sample_rbm_forward(visible_k, c, w)
    return hidden, visible_k

def spatial_actualise_weight(visible, b, c, w, k, epsilon, mode, Wt):
    hidden=sample_rbm_forward(visible, c, w)
    hidden_c, visible_c=backandforw(hidden, b, c, w, k, mode)
    w+=(1/(visible.shape[1]))*epsilon*(hidden.dot(np.transpose(visible))-hidden_c.dot(np.transpose(visible_c)))*Wt
    b+=(1/(visible.shape[1]))*epsilon*(np.sum(visible-visible_c, axis=1)).reshape(visible.shape[0],1)
    c+=(1/(visible.shape[1]))*epsilon*(np.sum(hidden-hidden_c, axis=1)).reshape(w.shape[0],1)#*g

def actualise_weight(visible, b, c, w, k, epsilon, mode):
    hidden=sample_rbm_forward(visible, c, w)
    hidden_c, visible_c=backandforw(hidden, b, c, w, k, mode)
    w+=(1/(visible.shape[1]))*epsilon*(hidden.dot(np.transpose(visible))-hidden_c.dot(np.transpose(visible_c)))
    b+=(1/(visible.shape[1]))*epsilon*(np.sum(visible-visible_c, axis=1)).reshape(visible.shape[0],1)
    c+=(1/(visible.shape[1]))*epsilon*(np.sum(hidden-hidden_c, axis=1)).reshape(w.shape[0],1)#*g

  
def train_rbm(visible, b, c, w, iterr_rbm, mode, epsilon, x_test, dataset_size, cd, f):
    ind=[]
    e=[]
    E=[]
    fe=[]
    fE=[]
    d=0
    g=0
    
    t1=0
    t2=0
    iterr_rbm=iterr_rbm//cd
    for i in range(iterr_rbm):
        visible_batch=visible[:,d:d+32]
        #visible_batch=visible_batch.reshape(visible.shape[0],1)
        g+=1
        if g==32:
            d+=32
            g=0
        if d>dataset_size-34:
            d=0
        actualise_weight(visible_batch, b, c, w, cd, epsilon, mode)
        if i%(iterr_rbm//20)==0:
            ind.append(i)
            if mode==0:
                tt=time.time()
                e.append(rbmv.energy_grbm(x_test, b, c, w))
                E.append(rbmv.energy_grbm(visible, b, c, w))
                t1+=time.time()-tt
                tt=time.time()
                fe.append(rbmv.pseudo_likelihood_grbm(x_test, b, c, w, f))
                #fE.append(rbmv.pseudo_likelihood_grbm(visible, b, c, w, 3))
                t2+=time.time()-tt
            else:
                tt=time.time()
                e.append(rbmv.energy_rbm(x_test, b, c, w))
                E.append(rbmv.energy_rbm(visible, b, c, w))
                t1+=time.time()-tt
                tt=time.time()
                fe.append(rbmv.pseudo_likelihood_rbm(x_test, b, c, w, f))
                #fE.append(rbmv.pseudo_likelihood_rbm(visible, b, c, w, 3))
                t2+=time.time()-tt

    plt.plot(ind, e,label="test set")
    plt.plot(ind, E, label="training set")
    plt.legend(loc='upper right')
    if mode == 1:
        plt.yscale('symlog')
    plt.show()
    plt.plot(ind, fe, label="free energy test set")
    #plt.plot(ind, fE, label="free energy training set")
    plt.legend(loc='lower right')
    plt.show()
    print("free-energy", fe[len(fe)-1])
    print("energy time and pseudo likelihood time ", t1, t2)

    
def train_spatial_rbm(visible, b, c, w, iterr_rbm, mode, epsilon, x_test, dataset_size, Wt):
    e=[]
    E=[]
    fe=[]
    fE=[]
    ind=[]
    t1=0
    t2=0
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
        spatial_actualise_weight(visible_batch, b, c, w, 1, epsilon, mode, Wt)
        if i%(iterr_rbm//20)==0:
            ind.append(i)
            if mode==0:
                tt=time.time()
                e.append(rbmv.energy_grbm(x_test, b, c, w))
                E.append(rbmv.energy_grbm(visible, b, c, w))
                t1+=time.time()-tt
                tt=time.time()
                fe.append(rbmv.pseudo_likelihood_grbm(x_test, b, c, w, 10))
                fE.append(rbmv.pseudo_likelihood_grbm(visible, b, c, w, 3))
                t2+=time.time()-tt
            else:
                tt=time.time()
                e.append(rbmv.energy_rbm(x_test, b, c, w))
                E.append(rbmv.energy_rbm(visible, b, c, w))
                t1+=time.time()-tt
                tt=time.time()
                fe.append(rbmv.pseudo_likelihood_rbm(x_test, b, c, w, 10))
                fE.append(rbmv.pseudo_likelihood_rbm(visible, b, c, w, 3))
                t2+=time.time()-tt
    """
    for i in range(iterr_rbm//3):
        actualise_weight(visible, b, c, w, i+1, 3, t)
        if i%50==0:
            e.append(energy(x_test, b, c, w))
            E.append(energy(x_train, b, c, w))    
    """
    plt.plot(ind, e,label="test set")
    plt.plot(ind, E, label="training set")
    plt.legend(loc='upper right')
    if mode == 1:
        plt.yscale('symlog')
    plt.show()
    plt.plot(ind, fe, label="free energy test set")
    plt.plot(ind, fE, label="free energy training set")
    plt.legend(loc='lower right')
    plt.show()
    print("energy", e[len(e)-1])
    print("energy time and pseudo likelihood time ", t1, t2)

