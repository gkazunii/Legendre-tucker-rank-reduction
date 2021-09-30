using TensorToolbox
using LinearAlgebra
using InvertedIndices
function TL1RR(T)
    input_tensor_depth = ndims(T)
    input_tensor_shape = size(T)

    partial_sums = []
    one_to_N = [1:input_tensor_depth;]
    for k=1:input_tensor_depth
        partial_sum = vec(sum(T, dims=one_to_N[Not(k)]))
        push!(partial_sums, partial_sum)
    end

    P = ttt(partial_sums[1], partial_sums[2])
    for n=3:input_tensor_depth
        P = ttt(P, partial_sums[n])
    end

    dev = sum(T)^(input_tensor_depth-1)
    P .= P ./ dev
    return P
end
