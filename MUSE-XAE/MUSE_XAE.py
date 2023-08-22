import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import sys
import pickle
import os
import argparse
from utils import load_dataset,plot_optimal_solution,plot_signature,train_model,optimal_model,calc_cosine_similarity,\
optimal_cosine_similarity,refit,plot_results,classification_from_representation
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    parser.add_argument('--iter', type=int, help='Number of repetitions for clustering',required=False,default=30)
    parser.add_argument('--max_sig', type=int, help='Max signatures to explore', default = 25, required=False)
    parser.add_argument('--min_sig', type=int, help='Min signatures to explore', default = 2, required=False)
    parser.add_argument('--augmentation', type=int, default=100,help='Number of data augmentation')
    parser.add_argument('--batch_size', type=int, default=64,help='Batch Size')
    parser.add_argument('--epochs', type=int, default=1000,help='Number of epochs')
    parser.add_argument('--mean_stability', type=int, help='Average Stability for accept a solution',default=0.7)
    parser.add_argument('--min_stability', type=int, help='Minimum Stability of a Signature to accept a solution',default=0.2)
    parser.add_argument('--directory', type=str, default='./', help='Main Directory to save results')
    parser.add_argument('--loss', type=str, default='poisson', help='Loss function to use in the autoencoder')
    parser.add_argument('--activation', type=str, default='softplus', help='activation function')
    parser.add_argument('--run', type=int, help='Parameter for multiple run to test robusteness')
    parser.add_argument('--n_jobs', type=int, help='number of cpu to use in parallel',required=False)



    args = parser.parse_args()
    data,iter,max_sig,min_sig=args.dataset,args.iter,args.max_sig,args.min_sig
    augmentation,batch_size,epochs=args.augmentation,args.batch_size,args.epochs
    mean_stability,min_stability,directory=args.mean_stability,args.min_stability,args.directory
    loss,activation,n_jobs=args.loss,args.activation,args.n_jobs

    if args.run : 
        iteration=args.run
        Main_dir=f'./Experiments/{directory}/{data}/Run_{iteration}'
        os.makedirs(Main_dir,exist_ok=True)
    else:
        Main_dir=f'./Experiments/{directory}/{data}'
        os.makedirs(Main_dir,exist_ok=True)

    Models_dir=f'{Main_dir}/Models/'
    os.makedirs(Models_dir,exist_ok=True)

    Plot_dir=f'{Main_dir}/Plots/'
    os.makedirs(Plot_dir,exist_ok=True)

    parameters = vars(args)

    with open(f'{Main_dir}/parameters.txt', 'w') as f:
        for key, value in parameters.items():
            f.write(f'{key}: {value}\n')

    # Load data
    X=load_dataset(name=data)
    
    # Signature extraction
    errors,extractions=optimal_model(X, iter=iter,max_sig=max_sig,min_sig=min_sig,loss=loss,batch_size=batch_size,epochs=epochs,augmentation=augmentation,activation=activation,
                                     save_to=Models_dir,n_jobs=n_jobs)

    min_cosine,mean_cosine,m_signatures,silhouettes=optimal_cosine_similarity(extractions,min_sig,max_sig)
    All_solutions_dir=f'{Main_dir}/All_Solutions/'
    os.makedirs(All_solutions_dir,exist_ok=True)

    for s in range(min_sig,max_sig+1):

        signatures_dir=f'{Main_dir}/All_Solutions/SBS96_{s}'
        os.makedirs(signatures_dir,exist_ok=True)

        sbs=pd.DataFrame(m_signatures[s])
        sbs.columns=[f'SBS_AE_{i+1}' for i in range(s)]
        sbs.to_csv(signatures_dir+'/SBS_96.csv')
        silh=pd.DataFrame(silhouettes[s])
        silh.index=sbs.columns
        silh.to_csv(signatures_dir+'/silhouettes.csv')

    mean_errors = {key: np.mean(values) for key, values in errors.items()}
    metrics = {'mean_errors': mean_errors, 'min_cosine': min_cosine, 'mean_cosine': mean_cosine}

    #Best solution
    df = pd.DataFrame.from_dict(metrics, orient='index').T.reset_index()
    df.columns=['signatures','mean_errors','min_cosine','mean_cosine']
    try:
        best=df[(df['mean_cosine']>=mean_stability) & (df['min_cosine']>=min_stability)].sort_values(by='mean_errors').iloc[0,:]
    except:
        best=df.sort_values(by='signatures').iloc[0,:]

    #Plot best solution
    plot_optimal_solution(save_to=Plot_dir,df_study=df,min_stability=min_stability,mean_stability=mean_stability)
    
    #Refit
    S=pd.DataFrame(m_signatures[best['signatures']])
    E=refit(X,S=S,best=best,save_to=Plot_dir)

            
    #Plot Signatures

    try:
        tumour_types=[column.split('::')[0] for column in X.index]
    except:
        try:
            tumour_types=[column.split('-')[0] for column in X.index]
        except:
            tumour_types=None
            
    index_signatures=X.columns
    plot_results(X,S=S,E=E,sig_index=index_signatures,tumour_types=tumour_types,save_to=Main_dir+'/')
