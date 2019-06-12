#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:03:45 2019

@author: gouraud
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.umath_tests import inner1d
import copy
import time

import rbm_train as rbmt
import function as func
ims=28

def energy_rbm(visible, b, c, w):
    e=0
    hidden=rbmt.sample_rbm_forward(visible, c, w)
    e+=-np.sum(b*visible)-np.sum(c*hidden)-np.sum(inner1d(hidden, w.dot(visible)))
    return e/(visible.shape[1]) 

def energy_grbm(visible, b, c, w):
    e=0
    hidden=rbmt.sample_rbm_forward(visible, c, w)
    e+=np.sum(np.power(b-visible, 2)/2)-np.sum(c*hidden)-np.sum(inner1d(hidden, w.dot(visible)))
    return e/(visible.shape[1]) 
    
def free_energy_rbm(visible, b, c, w):
    a=np.exp(w.dot(visible)+np.tile(c,(1,visible.shape[1])))
    return np.mean(-np.transpose(b).dot(visible)-np.sum(np.log(a+np.power(a,-1)), axis=0))#np.power(a,-1)

def free_energy_grbm(visible, b, c, w):
    a=np.exp(w.dot(visible)+np.tile(c,(1,visible.shape[1])))
    x=np.mean(np.sum(np.power(np.tile(b,(1,visible.shape[1]))-visible, 2), axis=0)/2-np.sum(np.log(a+np.power(a,-1)), axis=0))#np.power(a,-1)

    return  x

def flip(visible):
    i=np.random.randint(0, visible.shape[0])
    a=copy.deepcopy(visible)
    a[i,:]=-visible[i,:]
    return a

def pseudo_likelihood_rbm(visible, b, c, w, n):
    l=0
    for i in range(n):
        l+=np.log((func.sigmoid(free_energy_rbm(flip(visible),b,c,w)-free_energy_rbm(visible,b,c,w))+1)/2)
    return visible.shape[0]*l/n

def pseudo_likelihood_grbm(visible, b, c, w, n):
    l=0
    for i in range(n):
        l+=np.log((func.sigmoid(free_energy_grbm(flip(visible),b,c,w)-free_energy_grbm(visible,b,c,w))+1)/2)
    return visible.shape[0]*l/n


def gibbs_sampling(visible, b, c, w, n, scaler, mode):
    if mode[0]==0:
        plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(ims,ims), vmin=-100, vmax= 400)
        plt.colorbar()
        plt.show()
        a=visible
        for i in range(n):
            hidden=rbmt.sample_rbm_forward(a, c, w)
            visible=rbmt.sample_grbm_backward(hidden, b, w)
        plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(ims,ims), vmin=-100, vmax= 400)
        plt.colorbar()
        plt.show()
    else:
        plt.imshow(visible.reshape(ims,ims))
        plt.colorbar()
        plt.show()
        a=visible
        for i in range(n):
            hidden=rbmt.sample_rbm_forward(a, c, w)
            visible=rbmt.sample_rbm_backward(hidden, b, w)
        plt.imshow(visible.reshape(ims,ims))
        plt.colorbar()
        plt.show()        
    
def gibbs_deep_sampling(visible, b, c, w, n, scaler):
    #plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(ims,ims), vmin=-100, vmax= 400)
    #plt.colorbar()
    #plt.show()
    a=visible
    hidden=rbmt.sample_rbm_forward(a, c[0], w[0])
    for i in range(n):
        hidden=rbmt.sample_rbm_forward(hidden, c[i+1], w[i+1])
    for i in range(n):
        hidden=rbmt.sample_rbm_backward(hidden, b[n-i], w[n-i])
    visible=rbmt.sample_grbm_backward(hidden, b[0], w[0])
    plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(ims,ims), vmin=-100, vmax= 400)
    plt.colorbar()
    plt.show()
      
def fake_data(b, c, w, n):
    hidden=np.floor(1.1*np.random.random(size=c[n].shape))
    hidden, visible=rbmt.backandforw(hidden, b[n], c[n], w[n], 50, 1)
    N=n
    while N >1:
        visible=rbmt.sample_rbm_backward(visible, b[N-1], w[N-1])
        N-=1
    plt.imshow(rbmt.sample_grbm_backward(visible, b[N-1], w[N-1]).reshape(ims,ims))
    

#def error(visible, b,c,w):
#    return np.sum(np.abs(visible-rbmt.sample_rbm_backward(rbmt.sample_rbm_forward(visible,c,w),b,w)))/(visible.shape[0]*visible.shape[1])

def error(visible, b, c, w):
    a=visible
    return np.mean(np.abs(a-rbmt.sample_rbm_backward(rbmt.sample_rbm_forward(visible,c,w),b,w)))

def mean_pearson(visible, b, c, w): #compute the mean on test set of the pearson coefficient(image similitude) between an input image and it's reconstruction
    n=len(w)-1
    hidden=rbmt.sample_rbm_forward(visible, c[0], w[0])
    for i in range(n):
        hidden=rbmt.sample_rbm_forward(hidden, c[i+1], w[i+1])
    for i in range(n):
        hidden=rbmt.sample_rbm_backward(hidden, b[n-i], w[n-i])
    r_visible=rbmt.sample_grbm_backward(hidden, b[0], w[0])
    v_m=np.mean(visible, axis=0)
    r_v_m=np.mean(r_visible, axis=0)
    a=np.sum((r_visible-r_v_m)*(visible-v_m), axis=0)/(visible.shape[0]*np.std(r_visible, axis=0)*np.std(visible, axis=0))
    return np.mean(a), a