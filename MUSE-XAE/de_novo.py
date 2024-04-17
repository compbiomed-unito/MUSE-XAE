import pandas as pd
import numpy as np
import os
from data_preprocessing import load_dataset
from general_utils import optimal_model,optimal_cosine_similarity,refit
from data_visualization import plot_optimal_solution,plot_results,plot_exposures
import warnings
import os
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation




def denovo_extraction(args):

    print(' ')
    print('--------------------------------------------------')
    print(' ')
    print('         De Novo Extraction with MUSE-XAE')
    print(' ')
    print('--------------------------------------------------')  

    data,iter,max_sig,min_sig=args.dataset,args.iter,args.max_sig,args.min_sig
    augmentation,batch_size,epochs=args.augmentation,args.batch_size,args.epochs
    mean_stability,min_stability,directory=args.mean_stability,args.min_stability,args.directory
    loss,activation,n_jobs,cosmic_version=args.loss,args.activation,args.n_jobs,args.cosmic_version
    
    # refit parameters
    
    refit_only,reference_set,refit_patience,refit_penalty=args.refit_only,args.reference_set,args.refit_patience,args.refit_penalty
    remove_artefact,refit_regularizer,refit_loss=args.remove_artefact,args.refit_regularizer,args.refit_loss

    if args.run > 1 :
        iteration=args.run
        Main_dir=f'./Experiments/{directory}/{data}/De-Novo/Run_{iteration}'
        os.makedirs(Main_dir,exist_ok=True)
    else:
        Main_dir=f'./Experiments/{directory}/{data}/De-Novo/'
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
    X=load_dataset(name=data,cosmic_version=cosmic_version)

    # Signature extraction
    errors,extractions=optimal_model(X,iter=iter,max_sig=max_sig,min_sig=min_sig,loss=loss,batch_size=batch_size,epochs=epochs,augmentation=augmentation,activation=activation,n_jobs=n_jobs,
                                    save_to=Models_dir)

    min_cosine,mean_cosine,m_signatures,silhouettes=optimal_cosine_similarity(extractions,min_sig,max_sig)

    All_solutions_dir=f'{Main_dir}/All_Solutions/'
    os.makedirs(All_solutions_dir,exist_ok=True)

    for s in range(min_sig,max_sig+1):

        signatures_dir=f'{Main_dir}/All_Solutions/SBS96_{s}'
        os.makedirs(signatures_dir,exist_ok=True)

        sbs=pd.DataFrame(m_signatures[s])
        sbs.columns=[f'MUSE-SBS{chr(64+i+1)}' for i in range(s)]
        sbs.to_csv(signatures_dir+'/MUSE-SBS.csv')
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
    E=refit(X,S=S,best=best['signatures'],save_to=Models_dir)

            
    #Plot signatures
    try:
        tumour_types=[column.split('::')[0] for column in X.index]
    except:
        try:
            tumour_types=[column.split('-')[0] for column in X.index]
        except:
            tumour_types=None
            
    index_signatures=X.columns
    plot_results(X,S=S,E=E,sig_index=index_signatures,tumour_types=tumour_types,save_to=Main_dir+'/',cosmic_version=cosmic_version)
