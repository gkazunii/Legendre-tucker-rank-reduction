# About Code
The proposed algorithm is given in `TLrRR.jl`, which is implemented in Julia 1.6.1.
On the command line, we can use the algorithm as follows.
```
$ julia
julia> include("TLrRR.jl")
julia> X = rand(5,5,5)
julia> Y = TLrRR(X, [2,2,2])
```

The output `Y` is a (2,2,2)-rank tensor approximating `X`. 
We can confirm its Tucker rank by using `mrank` in `TensorToolbox`:
```
julia> using TensorToolbox
julia> mrank(Y)
(2, 2, 2)
```

Our experiments on real and synthetic datasets can be performed from the command line as follows.
```
$ julia main_real.jl
$ julia main_synthetic.jl
```

Results for synthetic and real datasets obtained by the above commands correspond to Fig. 3(a)(b) and Fig. 3(c)(d), respectively, in our paper.
Results will be saved in `../result` as jld2 files.
​
The following commands
```
cd plot
$ julia plot_realdata.jl
$ julia plot_syntheticadata.jl
```
make png images from jld2 files. The generated pdf files will be saved in `../png`.
We can modify experimental conditions and plot conditions by editing the file `config.jl`.
​

Real-world datasets are stored in `../data` as jld2 files. To access the files, `data_loader.jl` can be used.
​Please refer to appendix.pdf for the information on how to obtain these three datasets.