using FileIO
using Printf
include("config.jl")
include("calc_loader.jl")
include("utils/utils.jl")

function save_result(X, Xr, runtime, method, reqrank, cnt, fix)
    tensor_size = size(X)
    LSerror = norm(X-Xr)
    if method in non_negative_methods
        KLerror = get_dkl(X, Xr)
    else
        KLerror = missing
    end
    result = Dict(
                  "tensor_size" => tensor_size,
                  "runtime" => runtime,
                  "LSerror" => LSerror,
                  "KLerror" => KLerror,
                  "method" => method,
                  "reqrank" => reqrank
                 )
    size_str =  get_tensorinfo(tensor_size)
    reqrank_str =  get_tensorinfo(reqrank)
    if fix == "size"
        savepath = "../result/synthetic/size$size_str/reqrank$reqrank_str"
    elseif fix == "rank"
        savepath = "../result/synthetic/reqrank$reqrank_str/size$size_str"
    else
        error("improper $sz")
    end
    mkpath(savepath)
    save("$savepath/$method$cnt.jld2", result)
    return LSerror, KLerror
end

function get_random_tensor( tensor_size, random_tenosr_dist)
    if random_tenosr_dist == "uniform"
        X = rand(tensor_size...)
    elseif random_tenosr_dist == "gauss"
        X = abs.(randn(tensor_size...))
    end
    return X
end

function run_fix_size(tensor_size, reqranks)
    X = get_random_tensor(tensor_size, random_tenosr_dist)
    println("input tensor shape $tensor_size")
    println("__________________________________________")
    @printf("%5s %20s %8s %10s %10s %10s \n", "cnt", "reqrank", "method", "runtime", "LSerror", "KLerror")
    println("__________________________________________")
    for reqrank in reqranks
        for method in methods
            for cnt = 1:rep_max
                runtime = @elapsed begin
                    Xr = calc(X, method, reqrank)
                end
                LSerror, KLerror = save_result(X, Xr, runtime, method, reqrank, cnt, "size")
                @printf("%5s %20s %8s %10f %10f %10s \n", cnt, reqrank, method, runtime, LSerror, KLerror)
            end
        end
        println("__________________________________________")
    end
end

function run_fix_rank(tensor_sizes, reqrank)
    println("fixed reqrank $reqrank")
    println("__________________________________________")
    @printf("%5s %20s %8s %10s %10s %10s \n", "cnt", "tensor_size", "method", "runtime", "LSerror", "KLerror")
    println("__________________________________________")
    for tensor_size in tensor_sizes
        X = get_random_tensor(tensor_size, random_tenosr_dist)
        for method in methods
            for cnt = 1:rep_max
                runtime = @elapsed begin
                    Xr = calc(X, method, reqrank)
                end
                LSerror, KLerror = save_result(X, Xr, runtime, method, reqrank, cnt, "rank")
                @printf("%5s %20s %8s %10f %10f %10s \n", cnt, tensor_size, method, runtime, LSerror, KLerror)
            end
        end
        println("__________________________________________")
    end
end
d = 5
tensor_size = [30 for i=1:d]
reqranks = []
for n = 2:1:15
     reqrank = [n for i=1:d]
     push!(reqranks, reqrank)
end
run_fix_size(tensor_size, reqranks)

reqrank = [10,10,10]
tensor_sizes = [[30,30,30], [50,50,50], [60,60,60], [80,80,80],
               [100,100,100], [120,120,120], [150,150,150], [200,200,200], [300,300,300], [400,400,400], [500,500,500]]
run_fix_rank(tensor_sizes, reqrank)
