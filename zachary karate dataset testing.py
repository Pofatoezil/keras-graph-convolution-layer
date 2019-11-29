# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:40:36 2019

@author: harrylee
"""
from layers import GraphConv
from keras.layers import Input,Activation,Dense
from keras.models import Model
from networkx import to_numpy_matrix , karate_club_graph
import numpy as np
import matplotlib.pyplot as plt
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

in_feature=Input(shape=(None,34))
in_adj=Input(shape=(None,34))
gcn=GraphConv(2,activation='relu')([in_feature,in_adj])
output=Dense(1,activation='sigmoid')(gcn)

GCN=Model([in_feature,in_adj],output)
gcn_emb=Model([in_feature,in_adj],gcn)

GCN.summary()
GCN.compile(optimizer='Adam',loss='binary_crossentropy')

gcn_emb.predict([[X],[renormalization_trick]])

epochs=1000
loss=[]
emb_list=[]
for e in range(epochs):
    tmp_loss=0
    emb=None
    
    if e%100==0:
        emb=gcn_emb.predict([np.expand_dims(X,axis=0),
                         np.expand_dims(renormalization_trick,axis=0)])[0]
        emb_list.append(emb)
        plt.scatter(emb[:,0],emb[:,1],c=Y)
        plt.show()
    
    tmp_loss=GCN.train_on_batch([np.expand_dims(X,axis=0),
                         np.expand_dims(renormalization_trick,axis=0)],Y.reshape(1,34,1))
    loss.append(tmp_loss)
    print('epoch:{} , loss:{}'.format(e,tmp_loss))

