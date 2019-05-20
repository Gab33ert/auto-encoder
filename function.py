#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
"""
Created on Thu Apr 25 17:19:25 2019

@author: gouraud
"""
def sigmoid(x):
    return 1/(1+np.exp(-2*x))#(2/(1+np.exp(-x)))-1

def dsigmoid(x):
    f=sigmoid(x)
    return 2*f*(1-f)#-(f+1)*(f-1)/2

