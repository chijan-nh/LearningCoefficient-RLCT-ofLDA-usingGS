# LearningCoefficient-RLCT-ofLDA-usingGS


This is the numerical experimental `Julia 1.3.0` codes for calculating real log canonical threshold for latent Dirichlet allocation (LDA) using Gibbs sampling.  This experiment had been carried out for [1].

## Environment

Here, I show my environment used for the experiment.

### Hardware and OS

* CPU: Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
* RAM: 16.0 GB
* OS: Windows 10 Home 64bit

### Software

* Julia version 1.3.0 (later versions are maybe available but they are not tested).

For installation via `project.toml`, use the following:
```shell
cd thisrepo
julia --project=.
]
Pkg> instantiate
```

## Usage

### Using ipynb file

1. Open `Julia_calculate_RLCT_of_LDA_by_GS.ipynb` on Jupyter.

2. (If need) set parameters such the result storage location path, the size of matrix, the number of simulations, the sample size, hyperparameter, etc. in the third code cell.

3. Run all from the first cell.

### Using jl file

1. (If need) set parameters such the result storage location path, the size of matrix, the number of simulations, the sample size, hyperparameter, etc.

2. Run ```$ julia Julia_calculate_RLCT_of_LDA_by_GS.jl```.

## Ensure

Numerical calculation for the real log canonical threshold (RLCT a.k.a. learning coefficient) of LDA, via the Bayesian generalization error and WAIC averaged overall simulations, when the posterior distribution is realized by Gibbs sampling and the prior of the parameter stochastic matrix is uniform Dirichlet distribution.

## Contents

* `README.md`: this file.
* `Julia_calculate_RLCT_of_LDA_by_GS.ipynb`: the Jupyter Notebook file of the experiment code.
* `Julia_calculate_RLCT_of_LDA_by_GS.jl`: the Julia code file exported by the above ipynb file.
* `Project.toml`: the package file for instance installation packages.
* `Manifest.toml`: the package file for instance installation packages.
* `log/`: the default result storage directory.

## Reference

1. Naoki Hayashi. "The Exact Asymptotic Form of Bayesian Generalization Error in Latent Dirichlet Allocation", accepted at Neural Networks. The arXiv version is [here, arXiv: 2008.01304](https://arxiv.org/abs/2008.01304).


