#!/usr/bin/env python
# coding: utf-8
# %% [raw]
#

# %%

print('KERAS VERSION')
import sys
import datetime
print (sys.version)
import struct 
version = struct.calcsize("P")*8 
print(version)
import numpy as np
print(np.__version__)
from tensorflow.keras.callbacks import ModelCheckpoint
print('KERAS VERSION')
#import pip
#from pip._internal import main
#main(['install'] + ['--user'] + ['--upgrade'] +['pip'])
#package_names=['mat73', 'sklearn','tensorflow==2.4.0','keras==2.4.3']
#package_names=['mat73', 'sklearn']
#main(['install'] + ['--user'] + package_names + ['--upgrade']) 



print('KERAS VERSION')
import mat73
print('KERAS VERSION')
import sklearn
print('KERAS VERSION')
import keras 
print('KERAS VERSION')
print(keras.__version__)
from keras import layers
import tensorflow as tf
#print(tf.config.experimental.list_physical_devices('GPU'))
#print('TF VERSION')
#print(tf.__version__)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import scipy.io
import mat73

train_set = mat73.loadmat('train_set_parsed.mat')
train_set = train_set['train_set_parsed']
train_set[train_set<0] = 0
print(np.shape(train_set))
max_val = np.max(train_set)
min_val = np.min(train_set)

print('LOADED')

train_set = (train_set)/(max_val-min_val)
train_set = train_set[:,:,:,0:800];

vali_set = mat73.loadmat('vali_set_parsed_set_parsed_102.mat')
vali_set = vali_set['vali_set_parsed_102']

vali_set[vali_set<0] = 0
vali_set = (vali_set)/(max_val-min_val)
x_val = vali_set
x_val = x_val[:,:,:,0:800];



# Define constants
CLUSTER_NUM = 20
BATCH_SIZE = 58
UPDATE_NUM = 50
THRESHOLD = (13456/BATCH_SIZE) * UPDATE_NUM
EPOCHS = 500



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
# Some resources
# https://www.tutorialspoint.com/keras/keras_customized_layer.htm
# https://stackoverflow.com/questions/52031587/how-can-i-make-a-trainable-parameter-in-keras



#Define the graph/architecture

#from keras.optimizer_v2.adam import Adam as Adam
input_img = keras.Input(shape=(10, 10, 800))
x = layers.Conv2D(100, (10, 10), activation='relu', padding='same')(input_img)
x = layers.Conv2D(100, (10, 10), activation='relu', padding='same')(x)

u1_a = layers.Conv2D(50, (5, 5), activation='relu', padding='same')(x)
u1_a = layers.MaxPooling2D((2, 2), padding='same')(u1_a)
u1_b = layers.Conv2D(50, (5, 5), activation='relu', padding='same')(u1_a)
u1_c = layers.Flatten()(u1_b)
u1_encoded = layers.Dense(100, activation='relu', name= 'embedding')(u1_c)
u1_up_a = layers.Dense(1250, activation='relu')(u1_encoded)
u1_up_b = layers.Reshape((5,5,50))(u1_up_a)
u1_up_c = layers.Conv2D(50, (5, 5), activation='relu', padding='same')(u1_up_b)
u1_up_d = layers.UpSampling2D((2, 2))(u1_up_c)
u1_up_e = layers.Conv2D(50, (5, 5), activation='relu', padding='same')(u1_up_d)
u1_up_f = layers.Conv2D(50, (5, 5), activation='relu', padding='same')(u1_up_e)


#concatenated_embeddings = layers.concatenate([u1_encoded,u2_encoded,u3_encoded,u4_encoded,u5_encoded],axis = 1,name= 'embedding')
encoded = u1_encoded

#concatenated_decodes = layers.concatenate([u1_up_c,u2_up_c,u3_up_c,u4_up_c,u5_up_c],axis = 3)


cluster_output = cluster_layer()(encoded)
#cluster_layers = keras.Model(input_img, cluster_output)
#cluster_layers.compile(optimizer='adam', loss='kld')#binary_crossentropy
#cluster_layers.compile(optimizer='adam', loss='mse')#binary_crossentropy

x = layers.Conv2D(100, (5, 5), activation='relu', padding='same')(u1_up_f)

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

     hidden = self.cae.get_layer(name='embedding').output
     
               
     
     self.encoder = keras.Model(input_img, encoded)
     
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
     self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger,saver],validation_data=(x_val, x_val))
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

 def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):
     self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)
 
 # define training loop for ETE model 
 def fit(self, x, y=None, batch_size=BATCH_SIZE, maxiter=2e5, tol=1e-3, cae_weights=None, save_dir='./', validation_data = x_val):
    
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
     for ite in range(int(maxiter)):
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
             # Terminate once number of updates exceeds threshold     
             if ite > THRESHOLD:
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
ete.fit(train_set, y=train_set, tol=1e-3, maxiter=2e5, save_dir='./',
    cae_weights='model_140.hd5',validation_data=vali_set)
#model_2.hd5

# Acquire results on the training set from trained network
cluster_res = ete.predict(train_set)

mdic = {"cluster_out_train": cluster_res}
scipy.io.savemat("cluster_out_train.mat", mdic)

# Acquire reconstructed inputs to assess compression quality
decoded_imgs_train = ete.output_decoded(train_set)

mdic = {"decoded_imgs_train_a": decoded_imgs_train[:,:,:,0:400]}
scipy.io.savemat("decoded_imgs_train_a.mat", mdic)

mdic = {"decoded_imgs_train_b": decoded_imgs_train[:,:,:,400:]}
scipy.io.savemat("decoded_imgs_train_b.mat", mdic)




