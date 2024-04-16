import argparse

def parser_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    parser.add_argument('--iter', type=int, help='Number of repetitions for clustering',required=False,default=100)
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
    parser.add_argument('--cosmic_version',type=str,help='cosmic version for matching extracted signatures',default='3.4')
    parser.add_argument('--run', type=int, help='Parameter for multiple run to test robusteness',default=1)
    parser.add_argument('--n_jobs', type=int, help='number of cpu to use in parallel',required=False,default=12)
    parser.add_argument('--refit_only', type=str, help='Refit only a reference signatures set',required=False,default="False")
    parser.add_argument('--refit_patience', type=int,required=False,default=200)
    parser.add_argument('--refit_regularizer', type=str,required=False,default='l1')
    parser.add_argument('--refit_penalty', type=float,required=False,default=0.003)
    parser.add_argument('--refit_loss', type=str, default='mae', help='Refit Loss function to use in the autoencoder')
    parser.add_argument('--reference_set',type=str,default='COSMIC_SBS_GRCh37_3.4',required=False)
    parser.add_argument('--remove_artefact',type=str,required=False,default="True")
    
    args=parser.parse_args()

    return args
    