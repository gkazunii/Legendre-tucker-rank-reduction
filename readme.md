# Legendre Tucker-rank Reduction
Legendre Tucker-rank Reduction(LTR) is an efficient low-rank approximation algorithm for non-negative tensors. The current implementation supports any order tensors.

* Ghalamkari, K., Sugiyama, M. : **Fast Tucker Rank Reduction for Non-Negative Tensors Using Mean-Field Approximation**, NeurIPS 2021 (to appear). [[arXiv]](https://arxiv.org/abs/2103.02898) [[Slide]](https://mahito.nii.ac.jp/pdf/NeurIPS2021.pdf) [[Poster]](https://mahito.nii.ac.jp/pdf/Ghalamkari_NeurIPS2021_poster.pdf)

## Requirements
LTR is implemented in Julia 1.6.1.
We need only `src/LTR.jl` to run LTR.
`TensorToolBox` and `InvertedIndices` have to be installed in Julia.
All other files are for experiments in our paper.

## Usage
The proposed algorithm LTR is given in `src/LTR.jl`.
On the command line, we can use the algorithm as follows.
```julia
cd src
$ julia
julia> include("LTR.jl")
julia> X = rand(5,5,5)
julia> Y = LTR(X, [2,2,2])
```

The output `Y` is a (2,2,2)-rank tensor approximating `X`.
We can confirm its Tucker rank by using `mrank` in `TensorToolbox`:
```julia
julia> using TensorToolbox
julia> mrank(Y)
(2, 2, 2)
```

Note that `LTR(X, [1,1,1])` retruns the best rank-1 approximation of `X`, minimizing KL divergence from `X`.
You can find the algorithm of LTR in the appendix in the paper.

## Experiments in the paper
Our experiments on synthetic and real datasets can be performed from the command line as follows.
```julia
$ julia main_synthetic.jl
$ julia main_real.jl
```
You have to store real dataset `../data` as jld2 files in advance. To access the files, `data_loader.jl` can be used.
Please refer to the appendix in the paper for information on how to obtain these datasets.

Results for synthetic and real datasets obtained by the above commands correspond to Fig. 3(a)(b) and Fig. 3(c)(d), respectively, in our paper.
Results will be saved in `../result` as jld2 files.

The following commands
```julia
cd plot
$ julia plot_realdata.jl
$ julia plot_syntheticadata.jl
```
make png images from jld2 files. The generated png files will be saved in `../png`.
We can modify experimental conditions and plot conditions by editing the file `config.jl`.


## Citation
If you use LTR in a scientific publication, we would appreciate citations to the following paper:
* Ghalamkari, K., Sugiyama, M. : **Fast Tucker Rank Reduction for Non-Negative Tensors Using Mean-Field Approximation**, NeurIPS 2021 (to appear).

Bibtex entry:
```
@inproceedings{Ghalamkari2021FastTR,
  Title={Fast Tucker Rank Reduction for Non-Negative Tensors Using Mean-Field Approximation},
  Author={Ghalamkari, K. and Sugiyama, M.},
  Booktitle = {Advances in Neural Information Processing Systems 34},
  Month = {December},
  Year = {2021}
}
```

## Contact
Author: Kazu Ghalamkari  
Affiliation: National Institute of Informatics, Tokyo, Japan  
E-mail: gkazu@nii.ac.jp  
URL: [gkazu.info](http://gkazu.info)
