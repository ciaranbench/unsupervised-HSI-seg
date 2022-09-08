#!/usr/bin/env python
# coding: utf-8

import sys
import datetime
print (sys.version)
import struct 
import numpy as np
import mat73
import sklearn
import keras 
print(keras.__version__)
from keras import layers
import tensorflow as tf
import scipy.io

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Load and preprocess training/validation data
train_set = mat73.loadmat('train_set_parsed.mat')
train_set = train_set['train_set_parsed']
train_set[train_set<0] = 0

max_val = np.max(train_set)
min_val = np.min(train_set)


train_set = (train_set)/(max_val-min_val)
print('LOADED TRAINING SET')

vali_set = mat73.loadmat('vali_set_parsed.mat')
vali_set = vali_set['vali_set_parsed']
vali_set[vali_set<0] = 0
vali_set = (vali_set)/(max_val-min_val)
x_val = vali_set
print('LOADED VALIDATION SET')

# Define constants
CLUSTER_NUM = 3
BATCH_SIZE = 64
UPDATE_NUM = 10
THRESHOLD = (3136/BATCH_SIZE) * UPDATE_NUM
EPOCHS = 400

#Define custom cluster_layer for the clustering module. Code taken from https://github.com/XifengGuo/DCEC
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

#Sub-class the Layer class to define our custom cluster_layer
class cluster_layer(layers.Layer):
    # Initialise new class 
    def __init__(self, n_clusters=CLUSTER_NUM, alpha=1.0, **kwargs):
        super(cluster_layer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
    
    # Here, the layer is 'built' - trainable variables are defined. 
    def build(self, input_shape):
        # Create trainable weight variables for this layer.
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        super(cluster_layer, self).build(input_shape)
    
    # Define the operations performed by the layer 
    def call(self, inputs):
        """ Student t-distribution (t-SNE algorithm).
                 Compute q_ij = 1/(1+dist(x_i, u_j)^2), then normalise it.
        Arguments:
        inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    # This function outputs the layer's output shape. 
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

#Define the graph/architecture

input_img = keras.Input(shape=(20, 20, 800))
x = layers.Conv2D(100, (10, 10), activation='relu', padding='same')(input_img)
x = layers.Conv2D(100, (10, 10), activation='relu', padding='same')(x)

x1 = layers.Lambda(lambda x: x[:,:,:,0:20])(x) 
x2 = layers.Lambda(lambda x: x[:,:,:,20:40])(x) 
x3 = layers.Lambda(lambda x: x[:,:,:,40:60])(x) 
x4 = layers.Lambda(lambda x: x[:,:,:,60:80])(x) 
x5 = layers.Lambda(lambda x: x[:,:,:,80:100])(x) 

u1_a = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(x1)
u1_a = layers.MaxPooling2D((2, 2), padding='same')(u1_a)
u1_b = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u1_a)
u1_c = layers.Flatten()(u1_b)
u1_encoded = layers.Dense(20, activation='relu', name= 'embedding_1')(u1_c)
u1_up_a = layers.Dense(500, activation='relu')(u1_encoded)
u1_up_b = layers.Reshape((10,10,5))(u1_up_a)
u1_up_c = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u1_up_b)
u1_up_d = layers.UpSampling2D((2, 2))(u1_up_c)
u1_up_e = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u1_up_d)
u1_up_f = layers.Conv2D(20, (5, 5), activation='relu', padding='same')(u1_up_e)

u2_a = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(x2)
u2_a = layers.MaxPooling2D((2, 2), padding='same')(u2_a)
u2_b = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u2_a)
u2_c = layers.Flatten()(u2_b)
u2_encoded = layers.Dense(20, activation='relu', name= 'embedding_2')(u2_c)
u2_up_a = layers.Dense(500, activation='relu')(u2_encoded)
u2_up_b = layers.Reshape((10,10,5))(u2_up_a)
u2_up_c = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u2_up_b)
u2_up_d = layers.UpSampling2D((2, 2))(u2_up_c)
u2_up_e = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u2_up_d)
u2_up_f = layers.Conv2D(20, (5, 5), activation='relu', padding='same')(u2_up_e)

u3_a = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(x3)
u3_a = layers.MaxPooling2D((2, 2), padding='same')(u3_a)
u3_b = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u3_a)
u3_c = layers.Flatten()(u3_b)
u3_encoded = layers.Dense(20, activation='relu', name= 'embedding_3')(u3_c)
u3_up_a = layers.Dense(500, activation='relu')(u3_encoded)
u3_up_b = layers.Reshape((10,10,5))(u3_up_a)
u3_up_c = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u3_up_b)
u3_up_d = layers.UpSampling2D((2, 2))(u3_up_c)
u3_up_e = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u3_up_d)
u3_up_f = layers.Conv2D(20, (5, 5), activation='relu', padding='same')(u3_up_e)

u4_a = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(x4)
u4_a = layers.MaxPooling2D((2, 2), padding='same')(u4_a)
u4_b = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u4_a)
u4_c = layers.Flatten()(u4_b)
u4_encoded = layers.Dense(20, activation='relu', name= 'embedding_4')(u4_c)
u4_up_a = layers.Dense(500, activation='relu')(u4_encoded)
u4_up_b = layers.Reshape((10,10,5))(u4_up_a)
u4_up_c = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u4_up_b)
u4_up_d = layers.UpSampling2D((2, 2))(u4_up_c)
u4_up_e = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u4_up_d)
u4_up_f = layers.Conv2D(20, (5, 5), activation='relu', padding='same')(u4_up_e)

u5_a = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(x5)
u5_a = layers.MaxPooling2D((2, 2), padding='same')(u5_a)
u5_b = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u5_a)
u5_c = layers.Flatten()(u5_b)
u5_encoded = layers.Dense(20, activation='relu', name= 'embedding_5')(u5_c)
u5_up_a = layers.Dense(500, activation='relu')(u5_encoded)
u5_up_b = layers.Reshape((10,10,5))(u5_up_a)
u5_up_c = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u5_up_b)
u5_up_d = layers.UpSampling2D((2, 2))(u5_up_c)
u5_up_e = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(u5_up_d)
u5_up_f = layers.Conv2D(20, (5, 5), activation='relu', padding='same')(u5_up_e)

concatenated_embeddings = layers.concatenate([u1_encoded,u2_encoded],axis = 1)
concatenated_embeddings = layers.concatenate([concatenated_embeddings,u3_encoded],axis = 1)
concatenated_embeddings = layers.concatenate([concatenated_embeddings,u4_encoded],axis = 1)
concatenated_embeddings = layers.concatenate([concatenated_embeddings,u5_encoded],axis = 1,name= 'embedding')

encoded = concatenated_embeddings

concatenated_decodes = layers.concatenate([u1_up_f,u2_up_f],axis = 3)
concatenated_decodes = layers.concatenate([concatenated_decodes,u3_up_f],axis = 3)
concatenated_decodes = layers.concatenate([concatenated_decodes,u4_up_f],axis = 3)
concatenated_decodes = layers.concatenate([concatenated_decodes,u5_up_f],axis = 3)

cluster_output = cluster_layer()(encoded)

x = layers.Conv2D(100, (5, 5), activation='relu', padding='same')(concatenated_decodes)
decoded = layers.Conv2D(800, (5, 5), activation='relu', padding='same')(x)

# Use model subclassing to define custom training procedure.
# ETE -> 'End to end'
# https://keras.io/api/models/model/
# Modified code from https://github.com/XifengGuo/DCEC

# Import dependencies
from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans

# Define class. 
# Define autoencoder module that is pretrained in isolation. 
# 'Connect' autoencoder module to clustering module:
# Concat latent vectors produced from this pretrained autoencoder module
# Process latent vectors with clustering layer
# Train both clustering module and autoencoder together

# q is the output distribution
# p is the target distribution

class ETE(object):
 def __init__(self,
              input_shape,
              n_clusters=CLUSTER_NUM,
              alpha=1.0):

     super(ETE, self).__init__()

     self.n_clusters = n_clusters
     self.input_shape = input_shape
     self.alpha = alpha
     self.pretrained = False
     self.y_pred = []
    
     # Define encoder (used to extract latent vectors for CAE+k-means)
     self.encoder = keras.Model(input_img, encoded)
    
     # Define full autoencoder module
     self.cae = keras.Model(input_img, decoded)
     
     # Acquire latent vectors from autoencoder 
     hidden_1 = self.cae.get_layer(name='embedding_1').output
     hidden_2 = self.cae.get_layer(name='embedding_2').output
     hidden_3 = self.cae.get_layer(name='embedding_3').output
     hidden_4 = self.cae.get_layer(name='embedding_4').output
     hidden_5 = self.cae.get_layer(name='embedding_5').output
     
     # Concatenate latent vectors into single vector 
     concatenated_embeddings = keras.layers.concatenate([hidden_1,hidden_2],axis = 1)
     concatenated_embeddings = keras.layers.concatenate([concatenated_embeddings,hidden_3],axis = 1)
     concatenated_embeddings = keras.layers.concatenate([concatenated_embeddings,hidden_4],axis = 1)
     concatenated_embeddings = keras.layers.concatenate([concatenated_embeddings,hidden_5],axis = 1)   
        
     hidden = concatenated_embeddings
     
     # Connect autoencoder with clustering module -> ETE model
     clustering_layer = cluster_layer(name='clustering')(hidden)
     self.model = keras.Model(inputs=self.cae.input,
                        outputs=[clustering_layer, self.cae.output])
     
 # Function that pretrains the autoencoder module
 def pretrain(self, x, batch_size=BATCH_SIZE, epochs=EPOCHS, optimizer='adam', save_dir='./', validation_data = x_val):
     print('...Pretraining...')
     class CustomSaver(keras.callbacks.Callback):
         def on_epoch_end(self, epoch, logs={}):
            self.model.save("model_{}.hd5".format(epoch))
     
     self.cae.compile(optimizer=optimizer, loss='mse')
     #Log autoencoder module training progress
     from keras.callbacks import CSVLogger
     csv_logger = CSVLogger('pretrain_log.csv')
     saver = CustomSaver()
     # begin training
     self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger, saver],validation_data=(x_val, x_val))
     self.cae.save(save_dir + '/pretrain_cae_model')
     print('Pretrained weights are saved to %s/pretrain_cae_model' % save_dir)
     self.pretrained = True
     
 def load_weights(self, weights_path):
     self.model.load_weights(weights_path)
 
 def extract_feature(self, x):  # extract features from before clustering layer
     return self.encoder.predict(x)

 def predict(self, x):
     q, _ = self.model.predict(x, verbose=0)
     return q.argmax(1)
 
 # Function provides autoenoder output. 
 # Used to assess reconstruction quality and therefore a rough measure of 
 # how much 'useful' information encoded in latent vectors.
 def output_decoded(self, x):
     q, decod = self.model.predict(x, verbose=0)
     return decod

 @staticmethod
 def target_distribution(q):
     weight = q ** 2 / q.sum(0)
     return (weight.T / weight.sum(1)).T

 def compile(self, loss=['kld', 'mse'], loss_weights=[.1, 1], optimizer='adam'):
     self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)
 
 # define training loop for ETE model 
 def fit(self, x, y=None, batch_size=BATCH_SIZE, tol=1e-3, cae_weights=None, save_dir='./', validation_data = x_val):
    
     if not self.pretrained and cae_weights is None:
         print('...pretraining CAE using default hyper-parameters:')
         self.pretrain(x, batch_size, save_dir=save_dir,validation_data=x_val)
         self.pretrained = True
     elif cae_weights is not None:
         print('LOADING CAE WEIGHTS')
         self.cae.load_weights(cae_weights)
         print('cae_weights loaded successfully.')
     # Extract latent vectors after pretraining step to perform separate CAE+k-means procedure 
     encoded_imgs_pretrain = self.encoder.predict(train_set)
     mdic = {"encoded_imgs_pretrain": encoded_imgs_pretrain}
     scipy.io.savemat("encoded_imgs_pretrain.mat", mdic)
     encoded_imgs_pretrain = []
     mdic = []
     
     # Step 2: initialise cluster centres using k-means
     print('Initialising cluster centers with k-means.')
     kmeans = KMeans(n_clusters=self.n_clusters, n_init=CLUSTER_NUM)
     self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
     y_pred_last = np.copy(self.y_pred)
     self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
     print('Initialisation COMPLETE')
     
     # Step 3: deep clustering
     loss = [0, 0, 0]
     batch_index = 0
     for ite in range(int(THRESHOLD)):
         print('Training iteration: ' + str(ite))
         # Update target distribution at start, and after every epoch.
         # Both of these occur when batch_index = 0
         # (At end of epoch, batch index is reset to zero.) 
         
         if batch_index == 0:
             q, _ = self.model.predict(x, verbose=0) 
             _ = []
             # update the auxiliary target distribution p
             p = self.target_distribution(q)
             print('Target dist. updated') 

             # evaluate the clustering performance
             self.y_pred = q.argmax(1)
             
             # check stop criterion
             delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
             y_pred_last = np.copy(self.y_pred)
             
             if ite > 0 and delta_label < tol:
                 print('delta_label ', delta_label, '< tol ', tol)
                 print('Reached tolerance threshold. Stopping training.')
                 break
         
         # Determine whether an epoch has elapsed
         if (batch_index + 1) * batch_size > x.shape[0]:
             batch_index = 0
         # If not, update weights using batch
         else:
             loss = self.model.train_on_batch(x=x[batch_index * batch_size:(batch_index + 1) * batch_size],
                                              y=[p[batch_index * batch_size:(batch_index + 1) * batch_size],
                                                 x[batch_index * batch_size:(batch_index + 1) * batch_size]])
             batch_index += 1

     self.model.save_weights('./ete_model_final')

# Execute training
ete = ETE(input_shape=(20, 20, 800), n_clusters=CLUSTER_NUM)
ete.model.summary()
optimizer = 'adam'
ete.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=optimizer)
ete.fit(train_set, y=train_set, tol=1e-3, save_dir='./',
    cae_weights='model_65.hd5',validation_data=vali_set)

# Acquire results on the training set from trained network
cluster_res = ete.predict(train_set)

mdic = {"cluster_out_train": cluster_res}
scipy.io.savemat("cluster_out_train.mat", mdic)

# Acquire reconstructed inputs to assess compression quality
decoded_imgs_train = ete.output_decoded(train_set)

mdic = {"decoded_imgs_train": decoded_imgs_train}
scipy.io.savemat("decoded_imgs_train.mat", mdic)





