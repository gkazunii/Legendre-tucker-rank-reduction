using FileIO
using Printf
include("config.jl")
include("calc_loader.jl")
include("data_loader.jl")
include("utils/utils.jl")

function save_result(X, Xr, runtime, method, reqrank, cnt, datasetname)
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
    reqrank_str =  get_tensorinfo(reqrank)
    savepath = "../result/real/$datasetname/reqrank$reqrank_str"
    mkpath(savepath)
    save("$savepath/$method$cnt.jld2", result)
    return LSerror, KLerror
end


function run(datasetname, reqranks)
    X = load_tensor(datasetname)
    tensor_size = size(X)
    println("input tensor $datasetname shape $tensor_size")
    println("__________________________________________")
    @printf("%5s %10s %20s %8s %10s %10s %10s \n", "cnt", "datasetname", "reqrank", "method", "runtime", "LSerror", "KLerror")
    println("__________________________________________")
    for reqrank in reqranks
        for method in methods
            for cnt = 1:rep_max
                runtime = @elapsed begin
                    Xr = calc(X, method, reqrank)
                end
                LSerror, KLerror = save_result(X, Xr, runtime, method, reqrank, cnt, datasetname)
                @printf("%5s %10s %20s %8s %10f %10f %10s \n", cnt, datasetname, reqrank, method, runtime, LSerror, KLerror)
            end
        end
        println("__________________________________________")
    end
end

datasetname = "AttFace"
run(datasetname, plans[datasetname])
datasetname = "4DLFD"
run(datasetname, plans[datasetname])
