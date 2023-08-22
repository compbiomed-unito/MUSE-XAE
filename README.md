# MUtational Signature Extraction with an eXplainable AutoEncoder

![](Images/MUSE_XAE.png)

MUSE-XAE is a user-friendly tool powered by the robust capabilities of autoencoder neural networks, allowing for the extraction and visualization of mutational signatures present in a tumor catalog. MUSE-XAE consists of a hybrid denoising autoencoder with a nonlinear encoder that enables the learning of nonlinear interactions and a linear decoder that ensures interpretability. Based on the conducted experiments, MUSE-XAE has proven to be one of the most performant and accurate tools in extracting mutational patterns. To delve deeper into its workings, please read the paper

## Usage

We suggest to use MUSE-XAE with default parameters choosen based on experiments.

For Example to extract mutational signatures on the PCAWG datasets run the following:

python ./MUSE_XAE/MUSE_XAE.py --dataset PCAWG

The model gives the possibility to select the following arguments:

- `--dataset`: **(Required)** Dataset name.
- `--iter`: Number of repetitions for clustering. Default is `30`.
- `--max_sig`: Max signatures to explore. Default is `25`.
- `--min_sig`: Min signatures to explore. Default is `2`.
- `--augmentation`: Number of data augmentations. Default is `100`.
- `--batch_size`: Batch Size. Default is `64`.
- `--epochs`: Number of epochs. Default is `1000`.
- `--run`: Parameter for multiple runs to test robustness.
- `--mean_stability`: Average Stability for accepting a solution. Default is `0.7`.
- `--min_stability`: Minimum Stability of a Signature to accept a solution. Default is `0.2`.
- `--directory`: Main Directory to save results. Default is `./`.
- `--loss`: Loss function to use in the autoencoder. Default is `poisson`.
- `--activation`: Activation function. Default is `softplus`.
- `--n_jobs`: number of parallel jobs. Default is max_sig - min_sig.

## TO DO LIST
- update readme specifying input format and where to put datasets

