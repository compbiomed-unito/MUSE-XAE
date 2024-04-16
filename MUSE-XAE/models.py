import pandas as pd
import numpy as np
import os
from lap import lapjv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import activations,regularizers, constraints
from tensorflow.keras.constraints import NonNeg,Constraint
from tensorflow.keras.regularizers import OrthogonalRegularizer,L2
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Lambda,InputSpec,Layer,BatchNormalization,Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow.python.util.deprecation as deprecation


class KMeans_with_matching:
    def __init__(self, X, n_clusters, max_iter, random=False):
        
        self.X = X
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n = X.shape[0]

        if self.n_clusters > 1:
            model = KMeans(n_clusters=n_clusters, init='random').fit(self.X)
            self.C = np.asarray(model.cluster_centers_)
        else:
            self.C = self.X[np.random.choice(self.n, size=1), :]

        self.C_prev = np.copy(self.C)
        self.C_history = [np.copy(self.C)]    

    def fit_predict(self):
        
        if self.n_clusters == 1:
            cost = np.zeros((self.n, self.n))
            for j in range(self.n):
                cost[0,j] = 1 - cosine_similarity(self.X[j,:].reshape(1,-1), self.C[0,:].reshape(1,-1))
            for i in range(1, self.n):
                cost[i,:] = cost[0,:]
            _, _, colsol = lapjv(cost)
            self.partition = np.mod(colsol, self.n_clusters)
            self.C[0,:] = self.X[np.where(self.partition == 0)].mean(axis=0)
            return pd.DataFrame(self.C).T, self.partition

        for k_iter in range(self.max_iter):
            cost = np.zeros((self.n, self.n))
            for i in range(self.n_clusters): 
                for j in range(self.n):
                    cost[i,j]=1-cosine_similarity(self.X[j,:].reshape(1,-1),self.C[i,:].reshape(1,-1))
            for i in range(self.n_clusters, self.n):
                cost[i,:] = cost[np.mod(i,self.n_clusters),:]
            _,_,colsol = lapjv(cost)
            self.partition = np.mod(colsol, self.n_clusters)
            for i in range(self.n_clusters):
                self.C[i,:] = self.X[np.where(self.partition == i)].mean(axis=0)
            self.C_history.append(np.copy(self.C))
            if np.array_equal(self.C, self.C_prev):
                break
            else:
                self.C_prev = np.copy(self.C)
            
        return pd.DataFrame(self.C).T,self.partition


class minimum_volume(tf.keras.constraints.Constraint):
    def __init__(self, dim, beta):
        self.dim = dim
        self.beta = beta

    def __call__(self, weights):
        w_matrix = K.dot(weights, K.transpose(weights))
        det = tf.linalg.det(w_matrix + K.eye(self.dim))
        log_det = K.log(det) / K.log(10.0)
        return self.beta * log_det
    
    # Metodo per ottenere la configurazione della constraint
    def get_config(self):
        return {'dim': self.dim, 'beta': self.beta}

    # Metodo statico per creare una constraint a partire dalla sua configurazione
    @staticmethod
    def from_config(config):
        return minimum_volume(**config)

def MUSE_XAE(input_dim=96,l_1=128,z=17,beta=0.001,activation='softplus',reg='min_vol',refit=False,refit_regularizer='l1',refit_penalty=1e-3):

    """ hybrid autoencoder due to non linear encoder and linear decoder;
    NonNegativity constraint for the decoder """

    if reg=='min_vol': regularizer=minimum_volume(beta=beta,dim=z)
    elif reg=='ortogonal': regularizer=OrthogonalRegularizer(beta)
    elif reg=='L2' : regularizer=L2(beta)
    
    if refit==True:
        activation='relu'

    encoder_input=Input(shape=(input_dim,))
    
    latent_1 = Dense(l_1,activation='elu')(encoder_input)
    latent_1 = BatchNormalization()(latent_1)
    latent_1 = Dense(l_1/2,activation='elu')(latent_1)
    latent_1 = BatchNormalization()(latent_1)
    latent_1 = Dense(l_1/4,activation='elu')(latent_1)
    latent_1 = BatchNormalization()(latent_1)

    if refit==True: 
        if  refit_regularizer=='l1':
            signatures = Dense(z, activation=activation, activity_regularizer=regularizers.l1(refit_penalty) ,name='encoder_layer')(latent_1)
        else:
            signatures = Dense(z, activation=activation,activity_regularizer=regularizers.l2(refit_penalty), name='encoder_layer')(latent_1)

    else: 
        signatures = Dense(z,activation='softplus',name='encoder_layer')(latent_1)

    decoder  = Dense(input_dim,activation='linear',use_bias=False,kernel_constraint=NonNeg(),kernel_regularizer=regularizer)(signatures)
    encoder_model = Model(encoder_input,signatures)
    hybrid_dae = Model(encoder_input,decoder)
    
    return hybrid_dae,encoder_model

