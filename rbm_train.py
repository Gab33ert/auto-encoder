#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:56:41 2019

@author: gouraud
"""
import matplotlib.pyplot as plt
import numpy as np

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

  
def train_rbm(visible, b, c, w, iterr_rbm, mode, epsilon, x_test, dataset_size):
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
        actualise_weight(visible_batch, b, c, w, 1, epsilon, mode)
        if i%(iterr_rbm//20)==0:
            e.append(rbmv.energy(x_test, b, c, w))
            E.append(rbmv.energy(visible, b, c, w))
    """
    for i in range(iterr_rbm//3):
        actualise_weight(visible, b, c, w, i+1, 3, t)
        if i%50==0:
            e.append(energy(x_test, b, c, w))
            E.append(energy(x_train, b, c, w))    
    """
    plt.plot(e,label="test set")
    plt.plot(E, label="training set")
    plt.legend(loc='upper right')
    plt.yscale('symlog')
    plt.show()
    print("energy", e[len(e)-1])

    
def train_spatial_rbm(visible, b, c, w, iterr_rbm, mode, epsilon, x_test, dataset_size, Wt):
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
        spatial_actualise_weight(visible_batch, b, c, w, 1, epsilon, mode, Wt)
        if i%(iterr_rbm//20)==0:
            e.append(rbmv.energy(x_test, b, c, w))
            E.append(rbmv.energy(visible, b, c, w))
    """
    for i in range(iterr_rbm//3):
        actualise_weight(visible, b, c, w, i+1, 3, t)
        if i%50==0:
            e.append(energy(x_test, b, c, w))
            E.append(energy(x_train, b, c, w))    
    """
    plt.plot(e,label="test set")
    plt.plot(E, label="training set")
    plt.legend(loc='upper right')
    plt.yscale('symlog')
    plt.show()
    print("energy", e[len(e)-1])

