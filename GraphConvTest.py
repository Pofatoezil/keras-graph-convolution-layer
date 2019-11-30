# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:41:16 2019
to test GraphConv Layer

keras version 2.2.4 
tensorflow version 1.9.0 
numpy 1.15.1
perform perfectly 

but in tensorflow 1.13.1 (company environment), can't not use sparse matrix as input 

@author: WIN10
"""

import scipy.sparse as sp
from keras.layers import Layer , Input , Dense
import keras
from keras import backend as K
import tensorflow as tf
from keras.models import Model
import numpy as np

class GraphConv(Layer):#Z=Activation(AXW+b) , A=adjacency matrix, X=input feature ,W=weight b=bias
    """arXiv 1609.02907v4 Semi-supervised classification with graph convolution network method """
    def __init__(self, units, 
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 #activity_regularizer=None,
                 **kwargs):
        
        self.units=units  #轉換的feature 數量
        #self.step=step    #搜尋鄰近維度
        self.activation=keras.activations.get(activation)
        self.use_bias=use_bias
        self.kernel_initializer=keras.initializers.get(kernel_initializer)
        self.kernel_regularizer=keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint=keras.constraints.get(kernel_constraint)
        self.bias_initializer=keras.initializers.get(bias_initializer)
        self.bias_regularizer=keras.regularizers.get(bias_regularizer)
        self.bias_constraint=keras.constraints.get(bias_constraint)
        #self,activity_regularizer=keras.regularizers.get(activity_regularizer)
        self.support_masking=True   #使用mask 遮蔽0 
        self.W , self.b = None , None #initial weights
        super(GraphConv, self).__init__(**kwargs)
        
    def compute_mask(self, inputs, mask=None):
        if mask is None:
            mask = [None]
        return mask[0]
    
    def build(self,input_shape):
        feature_dim=int(input_shape[0][-1]) #input= [X,A] X=input feature , A=adjacency matrix
        self.W=self.add_weight(name='{}_W'.format(self.name),
                               shape=(feature_dim,self.units),
                               initializer=self.kernel_initializer,
                               regularizer=self.kernel_regularizer,
                               constraint=self.kernel_constraint
                               )
        if self.use_bias:
           self.b=self.add_weight(name='{}_b'.format(self.name),
                                  shape=(self.units,),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint
                                  )
        super(GraphConv,self).build(input_shape)
        
    def compute_output_shape(self,input_shape):
        #return input_shape[0][:2]+(self.units,)
        features_shape = input_shape[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)
    
    def call(self,inputs):
        X,A=inputs
#        A=K.cast(A,K.floatx()) #for np array
#        if isinstance(A,tf.SparseTensor):
#            feature=tf.sparse.matmul(A,X)
#        else:
#            feature=K.dot(A,X)
#        feature=K.dot(A,X)
#        feature=K.dot(feature,self.W)  #  AXW
        feature=K.dot(X,self.W)
        feature=K.dot(A,feature)
        if self.use_bias:
            feature+=self.b       #AXW+b
        return self.activation(feature) 
    

    
    def get_config(self,):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': keras.activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': keras.initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': keras.initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': keras.regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': keras.regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': keras.regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': keras.constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': keras.constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


input_A=sp.random(100,100 ,format='csr')
input_X=np.random.random((100,10))

model_input_A=Input(shape=(100,),sparse=True)
model_input_X=Input(shape=(10,))
gcn1=GraphConv(10,activation='relu')([model_input_X,model_input_A])
gcn2=GraphConv(10,activation='relu')([gcn1,model_input_A])
model=Model([model_input_X,model_input_A], gcn2)
model.summary()

ans=model.predict_on_batch([input_X,input_A])
ans.shape


def ClusterGCN(ft_length,gcn_layers,gcn_units,classes,activation=None):
    in_feature=Input(shape=(ft_length,),name='X')
    in_adj=Input(shape=(None,),name='A',sparse=True)
    
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

GCN=ClusterGCN(10,2,10,2)
GCN.predict_on_batch([input_X,input_A])



    
    
    