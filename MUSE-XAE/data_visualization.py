import pandas as pd
import numpy as np
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PyPDF2 import PdfMerger
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


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

    ax2.set_ylabel('Mean Silhouette', color=color)
    ax2.plot(df_study_sort['signatures'],df_study_sort['mean_cosine'],'o-',color=color);
    plt.xticks(np.arange(int(min(df_study_sort['signatures'])),int(max(df_study_sort['signatures']))+1))
    optimal_x = best['signatures']
    ax1.set_xlabel('Number of Signatures')
    plt.grid()
    plt.axvline(optimal_x, linestyle='--',color='grey')
    plt.title('Optimal Solution\n')
    plt.savefig(f'{save_to}Optimal_solution.pdf',bbox_inches='tight')


def plot_exposures(exposures,save_to):

    max_samples_per_plot = 60

    # Calcola il numero totale di pagine necessarie
    total_samples = exposures.shape[0]
    total_pages = np.ceil(total_samples / max_samples_per_plot).astype(int)

    # Crea un PDF per salvare i plot
    with PdfPages(f'{save_to}/Exposures_Signature.pdf') as pdf:
        for page in range(total_pages):
            # Seleziona un sottoinsieme di samples per l'attuale pagina
            start_idx = page * max_samples_per_plot
            end_idx = min((page + 1) * max_samples_per_plot, total_samples)
            subset = exposures.iloc[start_idx:end_idx]
            
            # Identifica le signature presenti (non zero) in questo subset
            non_zero_columns = subset.columns[(subset != 0).any(axis=0)]
            
            # Crea il plot
            fig, ax = plt.subplots(figsize=(12, 8))
            subset[non_zero_columns].plot(kind='bar', stacked=True, ax=ax)
            
            # Imposta i titoli e le etichette
            plt.xlabel('Samples')
            plt.ylabel('Number of SBS mutations')
        
            # Posiziona la legenda all'esterno del plot
            plt.legend(title='Signature', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Ajusta il layout e salva la pagina corrente nel PDF
            plt.tight_layout(rect=[0,0,0.85,1])  # Ajusta lo spazio per la legenda
            pdf.savefig(fig)
            plt.close(fig)

def plot_exposures_dist(exposures,save_to):

    data=exposures.copy()

    # Calcola il numero di samples non-zero per ogni signature
    non_zero_counts = (data != 0).sum()

    # Il numero totale di samples
    total_samples = data.shape[0]

    # Filtra le colonne (signatures) che sono presenti in almeno un sample
    filtered_columns = non_zero_counts[non_zero_counts > 0].index
    filtered_data = data[filtered_columns]

    # Prepara i dati per il box plot, considerando solo le signatures filtrate
    data_to_plot = [filtered_data[col] for col in filtered_columns]

    # Crea il plot
    fig, ax = plt.subplots(figsize=(15, 8))  # Larghezza dinamica basata sul numero di colonne

    # Genera i box plot con colori diversi
    bp = ax.boxplot(data_to_plot, patch_artist=True, vert=True, showfliers=True, positions=range(len(filtered_columns)))

    # Colora ogni box plot con un colore diverso
    colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_columns)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Imposta le etichette sull'asse x come vuote per rimuovere i numeri predefiniti
    ax.set_xticks(range(len(filtered_columns)))
    ax.set_xticklabels([''] * len(filtered_columns))  # Rimuove le etichette numeriche

    # Usa le frazioni come etichette sull'asse x
    for i, col in enumerate(filtered_columns):
        label = fr"$\frac{{{non_zero_counts[col]}}}{{{total_samples}}}$"
        ax.text(i, -0.02, label, ha='center', va='top', transform=ax.get_xaxis_transform(), rotation=0, fontsize=15)

    # Aggiungi le etichette delle signature sopra i box plot
    for i, col in enumerate(filtered_columns):
        ax.text(i, ax.get_ylim()[1], col, ha='center', va='bottom', rotation=45, fontsize=12)

    # Imposta i titoli degli assi
    ax.set_xlabel('')
    ax.set_ylabel('Number of SBS mutations')

    plt.tight_layout()
    plt.savefig(f'{save_to}Exposures_distribution.pdf',bbox_inches='tight')


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
    plot_exposures(E,save_to=Plot_dir)

    print(' ')
    print('Thank you for using MUSE-XAE! Check the results on the Experiments folder')
    print(' ')
