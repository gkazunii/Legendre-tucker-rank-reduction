using TensorToolbox
include("LTR.jl")
include("methods/NTD.jl")
include("methods/lraSNTD.jl")
include("methods/LD/LD.jl")

function tucker_to_cprank(tucker_rank)
    if all(isequal(first(tucker_rank)),tucker_rank)
        cp = tucker_rank[1]
    else
        error("rank $tucker_rank cannot convert to CP rank")
    end
    return cp
end

function calc(X, method, reqrank)
    if method == "LT1R"
        @assert tucker_to_cprank(reqrank) == 1
        return LT1R(X)
    elseif method == "LTR"
        return LTR(X, reqrank)
    elseif method == "NTD_KL"
        _, _, Xr = NTD(X, reqrank, cost="KL", init_method=NTD_init, verbose=false)
        return Xr
    elseif method == "NTD_LS"
        _, _, Xr = NTD(X, reqrank, cost="LS", init_method=NTD_init, verbose=false)
        return Xr
    elseif method == "lraSNTD"
        return lraSNTD(X, reqrank)
    else
        error("calc method error")
    end
end

