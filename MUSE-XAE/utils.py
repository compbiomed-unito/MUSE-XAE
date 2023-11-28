import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
import warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import h5py
import sys
import gc
import multiprocessing
from lap import lapjv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
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
from scipy.optimize import linear_sum_assignment
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger


tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


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
    def __init__(self, dim=15, beta=0.001):
        self.beta = beta
        self.dim = dim
    
    def __call__(self, weights):
        w_matrix = K.dot(weights, K.transpose(weights))
        det = tf.linalg.det(w_matrix + K.eye(self.dim))
        log_det = K.log(det)/K.log(10.0)
        return self.beta * log_det

    def get_config(self):
        return {'dim': self.dim,
                'beta': float(self.beta)}


def load_dataset(name='PCAWG',cosmic_version='3.4'):
    try:
        data=pd.read_csv(f'./datasets/{name}.csv')
    except:
        data=pd.read_csv(f'./datasets/{name}.txt',sep='\t')
    
    if cosmic_version=='3.4':

        COSMIC_sig=pd.read_csv('./datasets/COSMIC_SBS_GRCh37_3.4.txt',sep='\t').set_index('Type')
    else:
        COSMIC_sig=pd.read_csv('./datasets/COSMIC_SBS_GRCh37.txt',sep='\t').set_index('Type')

    data=data.set_index('Type')
    data=data.loc[COSMIC_sig.index].T
    
    return data


def data_augmentation(X,augmentation=5,augmented=True):

    X_augmented=[]
    
    for time in range(augmentation):
        X_bootstrapped=[]
        for x in X:
            N = int(round(np.sum(x)))
            p = np.ravel(x/np.sum(x))
            X_bootstrapped.append(np.random.multinomial(N, p))
        X_bootstrapped = np.array(X_bootstrapped)
        X_augmented.append(pd.DataFrame(X_bootstrapped))
    X_aug=pd.concat(X_augmented,axis=0)
        
    return X_aug


def base_plot_signature(array, axs, index, ylim=1):

    color = ((0.196,0.714,0.863),)*16 + ((0.102,0.098,0.098),)*16 + ((0.816,0.180,0.192),)*16 + \
            ((0.777,0.773,0.757),)*16 + ((0.604,0.777,0.408),)*16 + ((0.902,0.765,0.737),)*16
    
    color = list(color)

    width = max(array.shape)
    x = np.arange(width)
    if axs == None:
        f,axs = plt.subplots(1,figsize=(20,10))
    bars = axs.bar(x, array, edgecolor='black', color=color)

    plt.ylim(0, ylim)
    plt.yticks(fontsize=10)
    axs.set_xlim(-0.5, width) 
    axs.set_ylabel('Probability of mutation \n', fontsize=12)
    axs.set_xticks([])
    axs.set_xticks(x)  
    axs.set_xticklabels(index, rotation=90, fontsize=7)  


def plot_signature(signatures, name='DeNovo_Signatures', save_to='./'):

    index = ['A[C>A]A', 'A[C>A]C', 'A[C>A]G', 'A[C>A]T','C[C>A]A', 'C[C>A]C', 'C[C>A]G', 'C[C>A]T',
            'G[C>A]A', 'G[C>A]C', 'G[C>A]G', 'G[C>A]T', 'T[C>A]A', 'T[C>A]C', 'T[C>A]G', 'T[C>A]T',
            'A[C>G]A', 'A[C>G]C', 'A[C>G]G', 'A[C>G]T', 'C[C>G]A', 'C[C>G]C', 'C[C>G]G', 'C[C>G]T',
            'G[C>G]A', 'G[C>G]C', 'G[C>G]G', 'G[C>G]T', 'T[C>G]A', 'T[C>G]C', 'T[C>G]G', 'T[C>G]T',
            'A[C>T]A', 'A[C>T]C', 'A[C>T]G', 'A[C>T]T', 'C[C>T]A', 'C[C>T]C', 'C[C>T]G', 'C[C>T]T',
            'G[C>T]A', 'G[C>T]C', 'G[C>T]G', 'G[C>T]T', 'T[C>T]A', 'T[C>T]C', 'T[C>T]G', 'T[C>T]T',
            'A[T>A]A', 'A[T>A]C', 'A[T>A]G', 'A[T>A]T', 'C[T>A]A', 'C[T>A]C', 'C[T>A]G', 'C[T>A]T',
            'G[T>A]A', 'G[T>A]C', 'G[T>A]G', 'G[T>A]T', 'T[T>A]A', 'T[T>A]C', 'T[T>A]G', 'T[T>A]T', 
            'A[T>C]A', 'A[T>C]C', 'A[T>C]G', 'A[T>C]T', 'C[T>C]A', 'C[T>C]C', 'C[T>C]G', 'C[T>C]T',
            'G[T>C]A', 'G[T>C]C', 'G[T>C]G', 'G[T>C]T', 'T[T>C]A', 'T[T>C]C', 'T[T>C]G', 'T[T>C]T',
            'A[T>G]A', 'A[T>G]C', 'A[T>G]G', 'A[T>G]T', 'C[T>G]A', 'C[T>G]C', 'C[T>G]G', 'C[T>G]T',
            'G[T>G]A', 'G[T>G]C', 'G[T>G]G', 'G[T>G]T', 'T[T>G]A', 'T[T>G]C', 'T[T>G]G', 'T[T>G]T']

    n_signatures = signatures.shape[1]
    merger = PdfMerger()
    files_to_remove = []
    
    for signature in range(n_signatures):

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.set_style('darkgrid')
        s=signatures.loc[index].values[:, signature]
        base_plot_signature(s, axs=ax, index=index, ylim=max(s)+0.05)

        l1 = mpatches.Patch(color=(0.196, 0.714, 0.863), label='C>A')
        l2 = mpatches.Patch(color=(0.102, 0.098, 0.098), label='C>G')
        l3 = mpatches.Patch(color=(0.816, 0.180, 0.192), label='C>T')
        l4 = mpatches.Patch(color=(0.777, 0.773, 0.757), label='T>A')
        l5 = mpatches.Patch(color=(0.604, 0.777, 0.408), label='T>C')
        l6 = mpatches.Patch(color=(0.902, 0.765, 0.737), label='T>G')
        
        ax.text(0.01, 0.94, f'MUSE-SBS'+str(chr(64 +signature + 1))+'\n', transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
        
        #ax.set_title('SBS_AE_' + str(signature + 1)+'\n', fontsize=11, pad=20)
        ax.legend(handles=[l1, l2, l3, l4, l5, l6], loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.1),fontsize=18)

        file_name = f'{save_to}{name}_{signature+1}.pdf'
        plt.savefig(file_name, dpi=400)
        merger.append(file_name)
        files_to_remove.append(file_name)

    merger.write(f'{save_to}{name}.pdf')
    merger.close()
    
    # Removing the individual PDF files
    for file in files_to_remove:
        os.remove(file)

def normalize(X):

    total_mutations=X.sum(axis=1)
    total_mutations=pd.concat([total_mutations]*96,axis=1)
    total_mutations.columns=X.columns
    norm_data=X/total_mutations*np.log2(total_mutations)

    return np.array(norm_data,dtype='float64')


def plot_optimal_solution(save_to,df_study,min_stability,mean_stability):
    
    try:
        best=df_study[(df_study['mean_cosine']>=mean_stability) & (df_study['min_cosine']>=min_stability)].sort_values(by='mean_errors').iloc[0,:]
    except:
        best=df_study.sort_values(by='signatures').iloc[0,:]

    df_study_sort=df_study.sort_values(by='signatures')
    
    sns.set_style('darkgrid')

    fig, ax1 = plt.subplots(figsize=(10,8))
    color = 'tab:red'

    ax1.set_ylabel('Frobenius Norm', color=color)
    ax1.plot(df_study_sort['signatures'],df_study_sort['mean_errors'],'o-',color=color);

    ax2 = ax1.twinx()
    color = 'tab:blue'

    ax2.set_ylabel('Mean Cosine Similarity', color=color)
    ax2.plot(df_study_sort['signatures'],df_study_sort['mean_cosine'],'o-',color=color);
    plt.xticks(np.arange(int(min(df_study_sort['signatures'])),int(max(df_study_sort['signatures']))+1))
    optimal_x = best['signatures']
    ax1.set_xlabel('Number of Signatures')
    plt.grid()
    plt.axvline(optimal_x, linestyle='--',color='grey')
    plt.title('Optimal Solution\n')
    plt.savefig(f'{save_to}Optimal_solution.pdf',bbox_inches='tight')



def MUSE_XAE(input_dim=96,l_1=128,z=17,beta=0.001,activation='softplus',reg='min_vol',refit=False):

    """ hybrid autoencoder due to non linear encoder and linear decoder;
    NonNegativity constraint for the decoder """

    if reg=='min_vol': regularizer=minimum_volume(beta=beta,dim=z)
    elif reg=='ortogonal': regularizer=OrthogonalRegularizer(beta)
    elif reg=='L2' : regularizer=L2(beta)
    
    if refit==True:
        activation='relu'

    encoder_input=Input(shape=(input_dim,))
    
    latent_1 = Dense(l_1,activation=activation)(encoder_input)
    latent_1 = BatchNormalization()(latent_1)
    latent_1 = Dense(l_1/2,activation=activation)(latent_1)
    latent_1 = BatchNormalization()(latent_1)
    latent_1 = Dense(l_1/4,activation=activation)(latent_1)
    latent_1 = BatchNormalization()(latent_1)

    if refit==True: signatures = Dense(z, activation=activation,activity_regularizer=regularizers.l1(1e-3), name='encoder_layer')(latent_1)
    else: signatures = Dense(z,activation='softplus',name='encoder_layer')(latent_1)

    decoder  = Dense(input_dim,activation='linear',use_bias=False,kernel_constraint=NonNeg(),kernel_regularizer=regularizer)(signatures)
    encoder_model = Model(encoder_input,signatures)
    hybrid_dae = Model(encoder_input,decoder)
    
    return hybrid_dae,encoder_model


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


def refit(data,S,best,save_to='./'):
    print(' ')
    print('--------------------------------------------------')
    print(' ')
    print('   Assigning mutations to extracted signatures')
    print(' ')
    print('--------------------------------------------------')
    print('')
    
    original_data=np.array(data)
    X=normalize(data)
    
    model,encoder_model = MUSE_XAE(input_dim=96,z=int(best['signatures']),refit=True)
    S=S.apply(lambda x : x/sum(x))
    model.layers[-1].set_weights([np.array(S.T)])
    model.layers[-1].trainable=False 

    early_stopping=EarlyStopping(monitor='loss',patience=100)
    checkpoint=ModelCheckpoint(f'{save_to}best_model_refit.h5', monitor='loss', save_best_only=True, verbose=False)
    model.compile(optimizer=Adam(learning_rate=0.001),loss='mse',metrics=['mse','kullback_leibler_divergence'])
    history=model.fit(X,X,epochs=10000,batch_size=64,verbose=False,validation_data=(X,X),callbacks=[early_stopping,checkpoint])
      
    model_new=load_model(f'{save_to}best_model_refit.h5',custom_objects={"minimum_volume":minimum_volume(beta=0.001,dim=int(len(S.T)))})
    encoder_new = Model(inputs=model_new.input, outputs=model_new.get_layer('encoder_layer').output)    
    E=pd.DataFrame(encoder_new.predict(X))

    E = E.T.apply(lambda x: x / (sum(x) + 1e-10)) * np.array(original_data.sum(axis=1))
    E=E.T.round()
    
    return E

def plot_results(data,S,E,sig_index,tumour_types,save_to,cosmic_version):

    X=np.array(data)
    S=pd.DataFrame(S)
    E=pd.DataFrame(E)
    S.index=sig_index
    S=S.apply(lambda x : x/sum(x))
    S.columns=[ f'MUSE-SBS{chr(64+i+1)}' for i in range(0,S.shape[1])]
    E.columns=[ f'MUSE-SBS{chr(64+i+1)}' for i in range(0,E.shape[1])]

    Extraction_dir=f'{save_to}Suggested_SBS_De_Novo/'
    os.makedirs(Extraction_dir,exist_ok=True)

    S.to_csv(f'{Extraction_dir}MUSE_SBS.csv')
    E.index=tumour_types
    E.to_csv(f'{Extraction_dir}MUSE_EXP.csv')

    if cosmic_version=='3.4':
        
        COSMIC_sig=pd.read_csv('./datasets/COSMIC_SBS_GRCh37_3.4.txt',sep='\t').set_index('Type')
 
    else:
        COSMIC_sig=pd.read_csv('./datasets/COSMIC_SBS_GRCh37.txt',sep='\t').set_index('Type')
    
    cost=pd.DataFrame(cosine_similarity(S.T,COSMIC_sig.T))
    row_ind,col_ind=linear_sum_assignment(1-cost)
    reoreder_sig=S.iloc[:,row_ind]
    COSMIC=COSMIC_sig.iloc[:,col_ind]
    cosmic_match=pd.DataFrame([cosine_similarity(reoreder_sig.iloc[:,i].ravel().reshape(1,-1),COSMIC.iloc[:,i].ravel().reshape(1,-1))[0] for i in range(COSMIC.shape[1])],columns=['similiarity'])
    cosmic_match.insert(0,'MUSE-SBS',reoreder_sig.columns)
    cosmic_match.insert(1,'COSMIC-SBS',COSMIC.columns)
    cosmic_match.to_csv(f'{Extraction_dir}COSMIC_match.csv')
    
    Plot_dir=f'{save_to}Plots/'

    plot_signature(reoreder_sig,save_to=Plot_dir)
    
    print(' ')
    print('Thank you for using MUSE-XAE! Check the results on the Experiments folder')
    print(' ')


    
































