# MUtational Signatures Extraction with eXplainable AutoEncoder

![](Images/MUSE-XAE.png)

MUSE-XAE is a user-friendly tool powered by the robust capabilities of autoencoder neural networks, allowing for the extraction and visualization of SBS mutational signatures present in a tumor catalog. MUSE-XAE consists of a hybrid denoising autoencoder with a nonlinear encoder that enables the learning of nonlinear interactions and a linear decoder that ensures interpretability. Based on the experiments, MUSE-XAE has proven to be one of the best performing and accurate tools in extracting mutational signatures. To delve deeper into its workings, please read the related paper


## Instructions

After downloading the repository we suggest to create a conda environment with python 3.10 and 1.24.3 numpy consequently install the requirement libraries via pip, folliwing the step:

- Create the environment: `conda create -n MUSE_env python=3.10 numpy=1.24.3`

- Activate the environment: `conda activate MUSE_env `

- Installing other libraries: `pip install -r requirements.txt`

## Availability

MUSE-XAE is currently available through GitHub and the code runs only on CPU, 
but it will soon be available as an installable Python package and will be able to run on GPU


## Input

MUSE-XAE assumes that the input tumor catalog is in .csv o .txt (with tab separated) format.
The tumour catalogue `M` should be a `96xN` matrix where `N` is the number of tumours and `96` is the number of `SBS mutational classes`.
MUSE-XAE assumes that 96 mutational classes order is the one of `COSMIC`. If you want to use a different order in your catalogue please add a `Type` column with the desired order.
Finally put your `dataset` in the datasets folder. To have an idea of the input file structure you can find some examples in the `datasets` folder. 
In the following image we show the first five rows of the example dataset for simplicity. 

![](Images/Example_dataset.png)

All datasets reported in this repo and used in the paper are taken from `Uncovering novel mutational signatures by de novo extraction with SigProfilerExtractor` from Islam et al.

## Example Usage

For a quick test with the `Example` dataset run the following:

`python ./MUSE-XAE/MUSE_XAE.py --dataset Example --iter 5  --min_sig 2 --max_sig 5` 


## Complete Usage

We suggest to use MUSE-XAE with `default` parameters choosen based on experiments.
To extract mutational signatures with default parameters on the Example dataset run the following:

`python ./MUSE-XAE/MUSE_XAE.py --dataset Example`

The model gives also the possibility to select optional arguments:

- `--dataset`: **(Required)** Dataset name.
- `--augmentation`: Number of times of data augmentation. Default is `100`.
- `--iter`: Number of repetitions for clustering. Default is `100`.
- `--max_sig`: Max signatures to explore. Default is `25`.
- `--min_sig`: Min signatures to explore. Default is `2`.
- `--batch_size`: Batch Size. Default is `64`.
- `--epochs`: Number of epochs. Default is `1000`.
- `--run`: Parameter for multiple runs to test robustness.
- `--mean_stability`: Average Stability for accepting a solution. Default is `0.7`.
- `--min_stability`: Minimum Stability of a Signature to accept a solution. Default is `0.2`.
- `--directory`: Main Directory to save results. Default is `./`.
- `--loss`: Loss function to use in the autoencoder. Default is `poisson`.
- `--activation`: Activation function. Default is `softplus`.
- `--n_jobs`: number of parallel jobs. Default is `24`.
- `--cosmic_version`: Cosmic version reference. Default is `3.4` .


## Output

Running MUSE-XAE will generate an `Experiments` folder with different subfolders.
In `Plots` folder you can find signature plot (like the ones below) and in `Suggested_SBSB_DeNovo` you can find csv files for the extracted signatures and its relative exposures 
and the match with COSMIC database.

![](Images/Plot_signature.png)



