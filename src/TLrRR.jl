using StatsBase
include("TL1RR.jl")

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

function TLrRR(X, reqrank)
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

            X[idx...] = TL1RR( X[idx...] )
        end
    end
    return X
end
