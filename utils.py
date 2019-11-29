# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:19:39 2019

@author: harrylee
"""

import numpy as np
import time

def data_split(X,A,y,test_size=0.2):
    """
    Param X: np array , F features of N nodes, shape=(N,F)
    Param A: scipy sparse matrix , shape = (N,N) , notice that type(A.A) is np array
    Param y: np array , one hot label of nodes , shape=(N,L) , L is num of label
    Param test_size: float or int. If float, should be between 0.0 to 1.0  ,
                    represent the proportion of the dataset to include in the 
                    train split. If int, represent number of sample of test data
    Returns:
        
        X_train: np array , shape=(samples,F) , samples =num of trainning data
        A_train: scipy sparse matrix , shape(samples,samples)
        y_train: np array , shape=(samples,L)
        train_samples_idx: np array , lengths = samples , element in array means 
            the index of complete data
    """
    np.random.seed(int(time.time())) #random seed
    if isinstance(test_size,float):
        num_sample=int(len(X)*(1-test_size))
    elif isinstance(test_size,int):
        num_sample=len(X)-test_size
        
    train_samples_idx=np.random.choice(len(X),num_sample,replace=False)
    X_train=X[train_samples_idx]
    y_train=y[train_samples_idx]
    A_train=A[train_samples_idx][:,train_samples_idx]
    
    return X_train , A_train , y_train , train_samples_idx
    
        
    
    
    