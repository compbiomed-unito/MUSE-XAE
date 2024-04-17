import pandas as pd
import os
from general_utils import refit,consensus_refit
from data_preprocessing import load_dataset
from data_visualization import plot_exposures,plot_exposures_dist
import tensorflow.python.util.deprecation as deprecation
import tensorflow as tf


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
  
        Exposures_dir=f'{Main_dir}/Signature_Exposures/'
        os.makedirs(Exposures_dir,exist_ok=True)
        
        X=load_dataset(name=args.dataset,cosmic_version=args.cosmic_version)
        
        exposures=[]
        for n in range(5):
            E=refit(X,S=S,best=S.shape[1],save_to=Models_dir,refit_patience=args.refit_patience,
                refit_penalty=args.refit_penalty,refit_regularizer=args.refit_regularizer,refit_loss=args.refit_loss)
            E=E.reset_index(drop=True)

            exposures.append(E)
        
        consensus_exposures=consensus_refit(exposures)
        

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