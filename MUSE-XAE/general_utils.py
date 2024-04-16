import pandas as pd
import numpy as np
import multiprocessing
import sklearn
import tensorflow as tf
import os
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from models import MUSE_XAE,KMeans_with_matching,minimum_volume
from data_preprocessing import normalize,data_augmentation
from tensorflow.keras.optimizers.legacy import Adam



def train_model(data, signatures, iter, batch_size, epochs, loss,augmentation,activation,save_to):
        
    X_scaled=normalize(data)
    
    X_aug_multi_scaled=normalize(data_augmentation(X=np.array(data), augmentation=augmentation))

    model,encoder = MUSE_XAE(input_dim=96,z=signatures,activation=activation)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss=loss, metrics=['mse'])
    early_stopping=EarlyStopping(monitor='val_mse',patience=30)
    checkpoint=ModelCheckpoint(f'{save_to}best_model_{signatures}_{iter}.h5', monitor='val_mse', save_best_only=True, verbose=False)
    model.fit(X_aug_multi_scaled,X_aug_multi_scaled, epochs=epochs, batch_size=batch_size, verbose=False,validation_data=(X_scaled,X_scaled),callbacks=[early_stopping,checkpoint])
    model_new = load_model(f'{save_to}best_model_{signatures}_{iter}.h5',custom_objects={"minimum_volume":minimum_volume(beta=0.001,dim=int(signatures))})
    encoder_new = Model(inputs=model_new.input, outputs=model_new.get_layer('encoder_layer').output)
    
    S = model_new.layers[-1].get_weights()[0]
    E = encoder_new.predict(X_scaled)
    
    error = np.linalg.norm(np.array(X_scaled) - np.array(E.dot(S)))

    return error, S.T


def optimal_model(data, iter, min_sig, max_sig, loss , batch_size, epochs, augmentation, activation ,n_jobs, save_to='./'):
    
    # parallelize model training

    results = {}
    with multiprocessing.Pool(processes=n_jobs, maxtasksperchild=1) as pool:
        for signatures in range(min_sig, max_sig + 1):
            for i in range(iter):
                results[(signatures, i)] = pool.apply_async(train_model, (data, signatures, i, batch_size, epochs , loss, augmentation, activation, save_to))

        all_errors, all_extractions = {}, {}

        for signatures in range(min_sig, max_sig + 1):
            print('')
            print(f'Running {iter} iterations with {signatures} mutational signatures ...')
            errors, extractions = [], []
            for i in range(iter):
                error, S = results[(signatures, i)].get()
                errors.append(error)
                extractions.append(S)

            all_errors[signatures] = errors
            all_extractions[signatures] = extractions

    return all_errors, all_extractions


def calc_cosine_similarity(args):
    
    sig, all_extractions = args
    all_extraction_df = pd.concat([pd.DataFrame(df) for df in all_extractions[sig]], axis=1).T
    X_all = np.asarray(all_extraction_df)
    clustering_model = KMeans_with_matching(X=X_all, n_clusters=sig, max_iter=50)
    consensus_sig, cluster_labels = clustering_model.fit_predict()
    if sig==1:
        means_lst=[1]
        min_sil=1
        mean_sil=1

    else:
        sample_silhouette_values = sklearn.metrics.silhouette_samples(all_extraction_df, cluster_labels, metric='cosine')
        means_lst = []

        for label in range(len(set(cluster_labels))):
            means_lst.append(sample_silhouette_values[np.array(cluster_labels) == label].mean())
        min_sil=np.min(means_lst)
        mean_sil=np.mean(means_lst)
    
    return sig, min_sil,mean_sil,consensus_sig , means_lst


def optimal_cosine_similarity(all_extractions, min_sig=2,max_sig=15):

    pool = multiprocessing.Pool()
    results = pool.map(calc_cosine_similarity, [(sig, all_extractions) for sig in range(min_sig, max_sig+1)])

    min_cosine, mean_cosine,signatures,silhouettes = {}, {}, {}, {}
    for sig, min_val, mean_val, consensus_sig,means_lst in results:
        min_cosine[sig] = min_val
        mean_cosine[sig] = mean_val
        signatures[sig] = consensus_sig
        silhouettes[sig] = means_lst

    return min_cosine, mean_cosine,signatures,silhouettes
 

def refit(data,S,best,save_to='./',refit_patience=100,refit_penalty=1e-3,refit_regularizer='l1',refit_loss='mae'):
     
    original_data=np.array(data)
    X=normalize(data)
    model,encoder_model = MUSE_XAE(input_dim=96,z=int(best),refit=True,
                                   refit_penalty=refit_penalty,refit_regularizer=refit_regularizer)
    S=S.apply(lambda x : x/sum(x))
    model.layers[-1].set_weights([np.array(S.T)])
    model.layers[-1].trainable=False 

    early_stopping=EarlyStopping(monitor='val_mse',patience=refit_patience)
    checkpoint=ModelCheckpoint(f'{save_to}best_model_refit.h5', monitor='val_mse', save_best_only=True, verbose=False)
    model.compile(optimizer=Adam(learning_rate=0.0005),loss=refit_loss,metrics=['mse','kullback_leibler_divergence'])
    history=model.fit(X,X,epochs=10000,batch_size=128,verbose=False,validation_data=(X,X),callbacks=[early_stopping,checkpoint])
      
    model_new=load_model(f'{save_to}best_model_refit.h5',custom_objects={"minimum_volume":minimum_volume(beta=0.001,dim=int(len(S.T)))})
    encoder_new = Model(inputs=model_new.input, outputs=model_new.get_layer('encoder_layer').output)    
    E=pd.DataFrame(encoder_new.predict(X))

    sample_sum=original_data.sum(axis=1)
    d=E.apply(lambda x:x/sum(x)+1e-15,axis=1).reset_index(drop=True)
    d[d<0.05]=0
    d=d.apply(lambda x:x/sum(x)+1e-15,axis=1).reset_index(drop=True)
    E=d.mul(list(sample_sum),axis=0)
    E.columns=S.T.index
    E=E.round()
    E.fillna(0,inplace=True)

    return E


def consensus_refit(exposures, n_runs=5):

    print(' ')
    print('--------------------------------------------------')
    print(' ')
    print('   Assigning mutations to Signatures')
    print(' ')
    print('--------------------------------------------------')
    print('')

    consensus_matrix = pd.DataFrame(index=exposures[0].index, columns=exposures[0].columns)

    for index in consensus_matrix.index:
        rows = [df.loc[index] for df in exposures]
        
        masks = [row != 0 for row in rows]
        
        consensus_mask = sum(masks) > (len(exposures) / 2)
        
        for col in consensus_matrix.columns:
            if consensus_mask[col]:
                values = [row[col] for row, mask in zip(rows, masks) if mask[col]]

                if values:consensus_matrix.at[index, col] = np.median(values)
                else: consensus_matrix.at[index, col] = 0 
            else: consensus_matrix.at[index, col] = 0  

    return consensus_matrix




