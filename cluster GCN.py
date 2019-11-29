# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:42:21 2019
Cluster GCN

org_gcn_utils.py is from origin GCN author's github (https://github.com/tkipf/keras-gcn)
it's only used to get load data

gcn layer using layers.py , inputs [X,A] X.shape=(1,N,F) A.shape=(1,N,N)

@author: harrylee
"""
from org_gcn_utils import load_data
from utils import data_split
import scipy.sparse as sp 
import metis 
import networkx as nx
import numpy as np
from keras.models import Input , Model
from layers import GraphConv

GRAPH_DECOMPOSITION=20
GRAPH_PAR_ITER=10000
GC_LAYERS=2
GC_UNITS=100
GC_LAYERS_ACT='relu'
CLUSTERS_PER_BATCH=2
SPARSE_A=True


#load and spilt data
X,A,y =load_data(dataset="cora") #A is scipy spase matrix
X_train , A_train , y_train , train_samples=data_split(X,A,y,test_size=0.4)

def ClusterGCN(ft_length,gcn_layers,gcn_units,classes,activation=None):
    in_feature=Input(shape=(ft_length,),name='X')
    in_adj=Input(shape=(None,),name='A',sparse=SPARSE_A)
    
    #hidden gcn
    for _ in range(gcn_layers):
        if _ ==0:
            gcn=GraphConv(gcn_units,activation=activation,name='gcn_{}'.format(_))([in_feature,in_adj])
        else:
            gcn=GraphConv(gcn_units,activation=activation,name='gcn_{}'.format(_))([gcn,in_adj])
    out=GraphConv(classes,activation='softmax',name='out')([gcn,in_adj])
    model=Model([in_feature,in_adj],out)
    model.summary()
    return model

def MetisClustering(adj_matrix):
    G_train=nx.from_scipy_sparse_matrix(adj_matrix)

    _,cluster_out=metis.part_graph(G_train,nparts=GRAPH_DECOMPOSITION,niter=GRAPH_PAR_ITER)
    cluster_out=np.array(cluster_out)
    
    cluster_idx=[]
    for cluster in range(GRAPH_DECOMPOSITION):
        tmp_idx=np.where(cluster_out==cluster)
        cluster_idx.append(tmp_idx)
        del tmp_idx
    return cluster_out , cluster_idx

def MatrixExtract(X_train,A_train,y_train,cluster_idx,cluster_choice):
    """
    Param X_train , y_train:np array
    Param A_train , scipy sparse matrix
    Param cluster_idx: list, each element are tuple of cluster index
    Param cluster_choice: np.array , chosen cluster
    """
    r_stack_A=[] ;r_stack_X=[];r_stack_y=[]  
   
    for r,cluster_r in enumerate(cluster_choice):
        c_stack_A=[]
        
        for c,cluster_c in enumerate(cluster_choice):        
            part_A=A_train[cluster_idx[cluster_r][0]][:,
                          cluster_idx[cluster_c][0]].toarray()
            c_stack_A.append(part_A)
            
        column_concate_A=np.concatenate(c_stack_A,axis=1)
        r_stack_A.append(column_concate_A)
        r_stack_X.append(X_train[cluster_idx[cluster_r]])
        r_stack_y.append(y_train[cluster_idx[cluster_r]])
        
    combine_A=np.concatenate(r_stack_A,axis=0)
    combine_X=np.concatenate(r_stack_X,axis=0)
    combine_y=np.concatenate(r_stack_y,axis=0)
    return combine_X,combine_A,combine_y

def normalize(A):
    """
    Param A: np array ,Adjacency martix 
    Return:sp sparse matrix
    """
    diag_D=np.array(np.sum(A, axis=0)) 
    eye=np.eye(A.shape[0])
    A_hat=np.diag(1./(diag_D+1))*(A+eye)
    A_prime=A_hat+eye
    if SPARSE_A:
        A_prime=sp.csr_matrix(A_prime,copy=True)
    return A_prime

def data_gen(X_train, A_train , y_train , n_parts=GRAPH_DECOMPOSITION,
             n_choice=CLUSTERS_PER_BATCH):
    N,F=X_train.shape
    cluster_out , cluster_idx = MetisClustering(A_train)
    while True:
        cluster_choice=np.random.choice(n_parts,n_choice,replace=False)
        batch_X,batch_A,batch_y=MatrixExtract(X_train,A_train,y_train,cluster_idx,cluster_choice)
        norm_A=normalize(batch_A)
        
        #batch_X=np.expand_dims(batch_X,axis=0) #shape (batch , F) to (1,batch,F)
        #norm_A=np.expand_dims(norm_A,axis=0)   #shape (batch , batch) to (1,batch,batch)       
        yield [batch_X,norm_A] , batch_y

gen=data_gen(X_train,A_train , y_train)
CGCN=ClusterGCN(X_train.shape[1],2,100,y_train.shape[1],activation='relu')

pre , _=next(gen)
CGCN.predict(pre)
#CGCN.compile(optimizer='adam',loss='categorical_crossentropy')
#CGCN.fit_generator(gen,steps_per_epoch=100,epochs=20)
