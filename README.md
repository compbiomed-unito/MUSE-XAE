# MUtational Signature Extraction with eXplainable AutoEncoder

![](Images/MUSE_XAE.png)

MUSE-XAE is a user-friendly tool powered by the robust capabilities of autoencoder neural networks, allowing for the extraction and visualization of SBS mutational signatures present in a tumor catalog. MUSE-XAE consists of a hybrid denoising autoencoder with a nonlinear encoder that enables the learning of nonlinear interactions and a linear decoder that ensures interpretability. Based on the experiments, MUSE-XAE has proven to be one of the best performing and accurate tools in extracting mutational signatures. To delve deeper into its workings, please read the related paper


## Instructions

After downloading the repo we suggest to create a conda environment with the requirement libraries:

- `conda env create -f environment.yml -name env_name` 

Then activate the environment to Run MUSE_XAE:

- `conda activate env_name `


## Input

MUSE-XAE assumes that the input tumor catalog is in .csv o .txt (with tab separated) format.
The tumour catalogue `M` should be a `96xN` matrix where `N` is the number of tumours and `96` is the number of `SBS mutational classes`.
MUSE-XAE assumes that 96 mutational classes order is the one of `COSMIC`. If you want to use a different order in your catalogue please add a `Type` column with the desired order.
Finally put your `dataset` in the datasets folder. To have an idea of the input file structure you can find some examples in the `datasets` folder. 

## Usage

We suggest to use MUSE-XAE with `default` parameters choosen based on experiments.
To extract mutational signatures from the `Example` dataset run the following:

`python ./MUSE-XAE/MUSE_XAE.py --dataset Example`

The model gives also the possibility to select optional arguments:

- `--dataset`: **(Required)** Dataset name.
- `--augmentation`: Number of times of data augmentation. Default is `100`.
- `--iter`: Number of repetitions for clustering. Default is `30`.
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
- `--n_jobs`: number of parallel jobs. Default is max_sig - min_sig.

## Output

Running MUSE_XAE will generate an Experiments folder with some analysis and plots of your datasets


