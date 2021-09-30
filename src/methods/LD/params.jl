using Random
using InvertedIndices
using TensorToolbox

function rapid_tensor_from_theta(theta)
    tensor_shape = size(theta)
    d = ndims(theta)

    rs = []
    for k = 1:d
        theta_k = theta[ ntuple( j->j == k ? (:) : 1, d)... ]
        r = ones( tensor_shape[k] )
        r[1] = exp( theta_k[1] )
        for i = 2:tensor_shape[k]
            r[i] = r[i-1] * exp( theta_k[i] )
        end
        push!(rs, r)
    end

    tensor = ttt( vec(rs[1]), vec(rs[2]) )
    for k = 3:d
        tensor = ttt(tensor, vec(rs[k]) )
    end

    return tensor
end

function rapid_eta_from_tensor(T)
    tensor_shape = size(T)
    eta = zeros( tensor_shape )
    d = ndims(T)
    one_to_N = [1:d;]
    for i = 1:d
        tensor_shape_i = ones(Int, d)
        tensor_shape_i[i] = tensor_shape[i]
        tensor_shape_i_tp = tuple( tensor_shape_i... )
        sums = vec(sum(T, dims=one_to_N[Not(i)]))
        for idxes in reverse(Tuple(CartesianIndices( tensor_shape_i_tp )))
            eta[idxes] = sums[ idxes[i] ]
            if idxes[i] != tensor_shape[i]
                tmp = zeros(Int, d)
                tmp[i] = 1
                eta[idxes] += eta[ idxes + CartesianIndex( tuple(tmp...) ) ]
            end
        end
    end

    return eta
end

function rapid_theta_from_tensor(T)
    tensor_shape = size(T)
    theta = zeros( tensor_shape )
    d = ndims(T)
    for i = 1:d
        tensor_shape_i = ones(Int, d)
        tensor_shape_i[i] = tensor_shape[i]
        tensor_shape_i_tp = tuple( tensor_shape_i... )
        for idxes in CartesianIndices( tensor_shape_i_tp )
            theta[ idxes ] = log( T[ idxes ] )
            if idxes[i] > 1
                tmp = zeros(Int, d)
                tmp[i] = 1
                theta[ idxes ] -= log( T[ idxes - CartesianIndex( tuple(tmp...))] )
            end
        end
    end
    return theta
end

function get_uniform_eta( tensor_shape )
    eta = ones(tensor_shape)
    for idxes in CartesianIndices( tensor_shape )
        tmp = 1.0
        for i = 1:length(idxes)
            tmp *= (tensor_shape[i]-idxes[i]+1)
        end
        eta[ idxes ] = tmp
    end
    eta ./= prod(tensor_shape)
    return eta
end

function get_uniform_theta( tensor_shape )
    theta = zeros(tensor_shape)
    theta[1] = -1.0 * log(prod( tensor_shape ))
    return theta
end
