from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np


def conv(x, argdict):
    x = tf.layers.conv2d( 
                inputs      = x,
                filters     = argdict['filter'],
                kernel_size = argdict['kernel'],
                strides     = argdict['stride'],
                padding     = argdict['padding'], 
                use_bias    = argdict['bias'],
                name        = 'conv_'+str(argdict['nlayer']) )

    if argdict['BN']:
        # x = tf.layers.batch_normalization(inputs=x, momentum=0.9, training=argdict['train'], name='bnorm_'+str(argdict['nlayer']))
        with arg_scope([batch_norm],
                         scope='bnorm_'+str(argdict['nlayer']),
                         updates_collections=None,
                         decay=0.9,
                         center=True,
                         scale=True,
                         zero_debias_moving_mean=True):
            x = tf.cond(argdict['train'],
                lambda : batch_norm(inputs=x, is_training=argdict['train'], reuse=None),
                lambda : batch_norm(inputs=x, is_training=argdict['train'], reuse=True))


    if argdict['activate']:
        x = tf.nn.relu(features=x, name="relu_"+str(argdict['nlayer']))

    return x


def maxpool(x, argdict):
    x = tf.layers.max_pooling2d(
            inputs      = x,
            pool_size   = argdict['kernel'],
            strides     = argdict['stride'],
            padding     = argdict['padding'],
            name        = 'maxpool_'+str(argdict['nlayer'])
        )
    return x 

def fully_connected(x, argdict):
    x = tf.layers.dense(
            inputs      = x,
            units       = argdict['filter'],
            name        = 'fc_'+str(argdict['nlayer'])
        )
    if argdict['activate']:
        x = tf.nn.relu(x, name="relu_"+str(argdict['nlayer']))
    return x

def softmax(x, argdict):
    return tf.nn.softmax(x, name='softmax_'+str(argdict['nlayer']))

def dropout(x, argdict):
    '''
    only use in "big" model, should not use in VGG (paper version not included)
    L2 Loss has the similar function
    '''
    return tf.layers.dropout(inputs=x, rate=argdict['rate'], training=argdict['train'], name='dropout_'+argdict['nlayer'])


class VGG16():
    '''
    VGG-implement:
    basically follow the original version, but something different listed below [oeiginal setting]:
    1. batch_norm = False
    2. `dropout` do not necessary, but here is the optional choice
    '''
    def __init__(self, x, config, training, reuse=False):
        print(" >>>>>>>>> Backbone: VGG16 ")
        self.training        = training
        self.reuse           = reuse
        self.config          = config
        self.size            = self.config['train']['size']
        _cfile               = open(self.config['model']['dictionary']).readlines()
        self.classes         = []
        for line in _cfile:
            self.classes.append(line.strip())
        self.classes_num     = len(self.classes)
        self.model           = self.network(x)

    def network(self, x):
        with tf.variable_scope('VGG16', [x], reuse=self.reuse):
            ## pad for fixed size
            # x    = tf.pad(x, [[0, 0], [self.size, self.size], [self.size, self.size], [0, 0]])
            ## Block-1
            x1_1 = conv(x,       {'nlayer':1, 'filter':64, 'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x1_2 = conv(x1_1,    {'nlayer':2, 'filter':64, 'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x1_3 = maxpool(x1_2, {'nlayer':3,              'kernel':[2,2], 'stride':2, 'padding':'same'})
            ## Block-2
            x2_1 = conv(x1_3,    {'nlayer':4, 'filter':128,'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x2_2 = conv(x2_1,    {'nlayer':5, 'filter':128,'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x2_3 = maxpool(x2_2, {'nlayer':6,              'kernel':[2,2], 'stride':2, 'padding':'same'})
            ## Block-3
            x3_1 = conv(x2_3,    {'nlayer':7, 'filter':256,'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x3_2 = conv(x3_1,    {'nlayer':8, 'filter':256,'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x3_3 = conv(x3_2,    {'nlayer':9, 'filter':256,'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x3_4 = maxpool(x3_3, {'nlayer':10,             'kernel':[2,2], 'stride':2, 'padding':'same'})
            ## Block-4
            x4_1 = conv(x3_4,    {'nlayer':11,'filter':512,'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x4_2 = conv(x4_1,    {'nlayer':12,'filter':512,'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x4_3 = conv(x4_2,    {'nlayer':13,'filter':512,'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x4_4 = maxpool(x4_3, {'nlayer':14,             'kernel':[2,2], 'stride':2, 'padding':'same'})
            ## Block-5
            x5_1 = conv(x4_4,    {'nlayer':15,'filter':512,'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x5_2 = conv(x5_1,    {'nlayer':16,'filter':512,'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x5_3 = conv(x5_2,    {'nlayer':17,'filter':512,'kernel':[3,3], 'stride':1, 'padding':'same', 'bias':True, 'BN':True, 'activate':True, 'train':self.training})
            x5_4 = maxpool(x5_3, {'nlayer':18,             'kernel':[2,2], 'stride':2, 'padding':'same'})
            ## 3 FC layer
            x_fc1= fully_connected(x5_4,  {'nlayer':19, 'filter':4096,             'activate':True})
            x_fc1= flatten(x_fc1)
            x_fc2= fully_connected(x_fc1, {'nlayer':20, 'filter':4096,             'activate':True})
            x_fc3= fully_connected(x_fc2, {'nlayer':21, 'filter':self.classes_num, 'activate':False})
            ## last-layer: softmax
            # logit= softmax(x_fc3,        {'nlayer':22})

            ## summary
            print("=====================================================================")
            print("VGG-16 Network Structure:")
            print("x", np.shape(x))
            print("xpad", np.shape(x))
            print("x1_1\t", np.shape(x1_1))
            print("x1_2\t", np.shape(x1_2))
            print("x1_3\t", np.shape(x1_3))
            print("=====================================================================")
            print("x2_1\t", np.shape(x2_1))
            print("x2_2\t", np.shape(x2_2))
            print("x2_3\t", np.shape(x2_3))
            print("=====================================================================")
            print("x3_1\t", np.shape(x3_1))
            print("x3_2\t", np.shape(x3_2))
            print("x3_3\t", np.shape(x3_3))
            print("x3_4\t", np.shape(x3_4))
            print("=====================================================================")
            print("x4_1\t", np.shape(x4_1))
            print("x4_2\t", np.shape(x4_2))
            print("x4_3\t", np.shape(x4_3))
            print("x4_4\t", np.shape(x4_4))
            print("=====================================================================")
            print("x5_1\t", np.shape(x5_1))
            print("x5_2\t", np.shape(x5_2))
            print("x5_3\t", np.shape(x5_3))
            print("x5_4\t", np.shape(x5_4))
            print("=====================================================================")
            print("x_fc1\t", np.shape(x_fc1))
            print("x_fc2\t", np.shape(x_fc2))
            print("x_fc3\t", np.shape(x_fc3))
            print("logit\t", np.shape(logit))
            print("=====================================================================")
            
            return logit





