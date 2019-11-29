# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:31:51 2019

@author: harrylee
"""
import keras
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf

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

if __name__ == '__main__':
    #test layer
    import numpy as np
    input_data = [
        [
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
            [7, 7, 8],
        ]
    ]
        
    input_edge = [
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ]
    
    input_data=np.array(input_data)
    input_edge=np.array(input_edge)
    input_data2=np.squeeze(input_data)
    input_edge2=np.squeeze(input_edge)
    
    in_feature=keras.layers.Input(shape=(None, 3), name='Input-Data')
    in_edge=keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
    gcn=GraphConv(2,kernel_initializer='ones',bias_initializer='ones',
                  name='GraphConv')([in_feature,in_edge])
    model = keras.models.Model(inputs=[in_feature, in_edge], outputs=gcn)
    model.compile(optimizer='adam',loss='mae',metrics=['mae'])
    model.summary()
    predicts = model.predict([input_data, input_edge])[0]
    predicts2=model.predict([[input_data2],[input_edge2]])    
#    ans
#    [[28,28],
#     [13,13],
#     [19,19],
#     [23,23]]
