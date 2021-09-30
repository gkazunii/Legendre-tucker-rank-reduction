using TensorToolbox
using LinearAlgebra
using InvertedIndices
using StatsBase

"""
This function gives the best rank-1 approximation of the non-negative tensor `T`
"""
function LT1R(T)
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


function get_basis_positions(tensor_size, reqrank)
    # Non bingo position become basis for reconstration tensor.
    # Basis position is non-bingo posision
    D = length(reqrank)
    basis_positions = []
    for d = 1 : D
        basis_position = [1; sample(2:tensor_size[d], reqrank[d]-1, replace=false, ordered=true)]
        push!(basis_positions, basis_position)
    end
    return basis_positions
end

"""
Legendre Tucker-rank Reduction,
proposed by Kazu Ghalamkari and Mahito Sugiyama.

# Aruguments
- `X`: input non-negative tensor, multi dimensional array.
- `reqrank` : target Tucker rank, array.
"""
function LTR(X, reqrank)
    X = copy(X)
    @assert ndims(X) == length(reqrank)
    D = ndims(X)
    tensor_size = size(X)
    basis_positions = get_basis_positions(tensor_size, reqrank)

    for d=1:D
        for b_number = 1 : reqrank[d]
            rank1_subtensor_begin = basis_positions[d][b_number]
            if b_number != reqrank[d]
                rank1_subtensor_end = basis_positions[d][b_number+1] - 1
            else
                rank1_subtensor_end = tensor_size[d]
            end

            if rank1_subtensor_begin == rank1_subtensor_end
                continue
            end
            idx = ntuple( l -> l == d ? (rank1_subtensor_begin:rank1_subtensor_end) : (:), D)

            X[idx...] = LT1R( X[idx...] )
        end
    end
    return X
end
