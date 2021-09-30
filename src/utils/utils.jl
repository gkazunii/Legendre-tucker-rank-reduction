function get_tensorinfo(tensor_shape)
    tensor_info = ""
    tensor_info_for_title = ""
    for s in tensor_shape
        tensor_info = tensor_info*"_"*string(lpad(s, n_digits, "0"))
    end
    return tensor_info
end

function get_dkl(P, Q ; eps_thred = 1.0e-10)
    tensor_shape = size(P)
    dkl = 0.0
    for idx in CartesianIndices(tensor_shape)
        if P[idx] < eps_thred
            continue
        end
        if Q[idx] < eps_thred
            continue
        end
        dkl += P[idx] * log( P[idx] / Q[idx] )
    end
    return dkl - sum(P) + sum(Q)
    #return sum( P .* log.( P ./ Q ) ) - sum(P) + sum(Q)
end
