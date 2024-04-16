import pandas as pd
import numpy as np


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


def normalize(X):

    total_mutations=X.sum(axis=1)
    total_mutations=pd.concat([total_mutations]*96,axis=1)
    total_mutations.columns=X.columns
    norm_data=X/total_mutations*np.log2(total_mutations)

    return np.array(norm_data,dtype='float64')

