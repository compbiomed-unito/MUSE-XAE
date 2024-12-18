import pandas as pd
import os
from general_utils import refit,consensus_refit,refit_process,row_wise_cosine_similarity,l2_norm,row_wise_pearson_similarity,compute_statistics
from data_preprocessing import load_dataset
from data_visualization import plot_exposures,plot_exposures_dist
import tensorflow.python.util.deprecation as deprecation
import tensorflow as tf
import multiprocessing


def signature_assignment(args):

        print(' ')
        print('--------------------------------------------------')
        print(' ')
        print('         Refit with MUSE-XAE')
        print(' ')
        print('--------------------------------------------------')


        try:
            S=pd.read_csv(f'./datasets/{args.reference_set}.csv').set_index('Type')
        except:
            S=pd.read_csv(f'./datasets/{args.reference_set}.txt',sep='\t').set_index('Type')


        if args.remove_artefact=='True':

            artefact_3_3=['SBS27', 'SBS43', 'SBS45', 'SBS46', 'SBS47', 'SBS48', 'SBS49', 'SBS50', 'SBS51', 'SBS52', 'SBS53', 'SBS54',
            'SBS55', 'SBS56', 'SBS57', 'SBS58', 'SBS59', 'SBS60'] 

            artefact_3_4=['SBS27', 'SBS43', 'SBS45', 'SBS46', 'SBS47', 'SBS48', 'SBS49', 'SBS50', 'SBS51', 'SBS52', 'SBS53', 'SBS54',
            'SBS55', 'SBS56', 'SBS57', 'SBS58', 'SBS59', 'SBS60','SBS95']
            
            try:
                S=S[S.columns.drop(artefact_3_4)]
            except:
                S=S[S.columns.drop(artefact_3_3)]


        Main_dir=f'./Experiments/{args.directory}/{args.dataset}/Refit'
        os.makedirs(Main_dir,exist_ok=True)
        
        Models_dir=f'{Main_dir}/Models/'
        os.makedirs(Models_dir,exist_ok=True)

        Plot_dir=f'{Main_dir}/Plots/'
        os.makedirs(Plot_dir,exist_ok=True)
  
        Stats_dir=f'{Main_dir}/Statistics/'
        os.makedirs(Stats_dir,exist_ok=True)

        Exposures_dir=f'{Main_dir}/Signature_Exposures/'
        os.makedirs(Exposures_dir,exist_ok=True)

        parameters = vars(args)

        with open(f'{Main_dir}/parameters.txt', 'w') as f:
            for key, value in parameters.items():
                f.write(f'{key}: {value}\n')
    

        X=load_dataset(name=args.dataset,cosmic_version=args.cosmic_version)
    
    
        with multiprocessing.Pool(10) as pool:
            args_list = [(X, S, Models_dir, args.refit_patience, args.refit_penalty, args.refit_regularizer, args.refit_loss, args.batch_size, r) for r in range(10)]
            exposures = pool.starmap(refit_process, args_list)
        
        consensus_exposures=consensus_refit(exposures,X)
        

        try:
            tumour_types=[column.split('::')[0] for column in X.index]
        except:
            try:
                tumour_types=[column.split('-')[0] for column in X.index]
            except:
                tumour_types=None
        
        consensus_exposures.index=tumour_types
        consensus_exposures.to_csv(f'{Exposures_dir}MUSE_EXP.csv')

        plot_exposures(consensus_exposures,save_to=Plot_dir)
        plot_exposures_dist(consensus_exposures,save_to=Plot_dir)

        compute_statistics(X,consensus_exposures,S,Stats_dir)

        print(' ')
        print('Thank you for using MUSE-XAE! Check the results on the Experiments folder')
        print(' ')
