# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:40:36 2019

@author: harrylee
"""
from layers_vr2 import GraphConv
from keras.layers import Input,Activation,Dense
from keras.models import Model
from networkx import to_numpy_matrix , karate_club_graph
import numpy as np
import matplotlib.pyplot as plt
import copy

zkc = karate_club_graph()
order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())

#make labe
Y=np.zeros(len(A))
for i in zkc._node:
    if zkc._node[i]['club']=='Mr. Hi':
        Y[i]=0
    else:
        Y[i]=1
        
A_hat=A+I
D_= np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_))

renormalization_trick=np.diag(1./np.sqrt(D_))*A_hat*np.diag(1./np.sqrt(D_))
X=I
#--------------------------------------------------------------------------

in_feature=Input(shape=(34,))
in_adj=Input(shape=(34,))
gcn=GraphConv(2,activation='relu')([in_feature,in_adj])
output=Dense(1,activation='sigmoid')(gcn)

GCN=Model([in_feature,in_adj],output)
gcn_emb=Model([in_feature,in_adj],gcn)

GCN.summary()
GCN.compile(optimizer='Adam',loss='binary_crossentropy')

#gcn_emb.predict_on_batch([X,renormalization_trick])

def trainning_test():
    epochs=1000
    loss=[]
    emb_list=[]
    for e in range(epochs):
        tmp_loss=0
        emb=None
        
        if e%100==0:
            emb=gcn_emb.predict_on_batch([X,renormalization_trick])
            emb_list.append(emb)
            plt.scatter(emb[:,0],emb[:,1],c=Y)
            plt.show()
        
        tmp_loss=GCN.train_on_batch([X,renormalization_trick],Y.reshape(34,1))
        loss.append(tmp_loss)
    return loss, emb_list

def predicting_test():
    rand_list=np.random.rand(34)
    train_mask=rand_list>0.4
    predict_mask=rand_list<0.4
    epochs=1000
    loss=[]
    emb_list=[]
    acc=[]
    for e in range(epochs):
        tmp_loss=0
        emb=None
        
        if e%100==0:
            emb=gcn_emb.predict_on_batch([X,renormalization_trick])
            emb_list.append(emb)
            C=copy.copy(Y)
            mask0=Y==0.
            mask1=Y==1.
            C[train_mask & mask0]=3.
            C[train_mask & mask1]=4.
            plt.scatter(emb[:,0],emb[:,1],c=C)
            plt.show()
        tmp_loss=GCN.train_on_batch([X,renormalization_trick],Y.reshape(34,1),
                                    sample_weight=train_mask)
        loss.append(tmp_loss)
        
        predicts=GCN.predict_on_batch([X,renormalization_trick])[predict_mask]
        predicts[predicts<0.5]=0. 
        predicts[predicts>0.5]=1.
        accuracy=sum(predicts[:,0]==Y[predict_mask][:])/len(predicts)
        acc.append(accuracy)
         
    return loss, emb_list ,acc

loss, emb_list, final_acc = predicting_test()