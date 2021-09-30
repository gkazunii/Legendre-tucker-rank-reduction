using TensorToolbox
using LinearAlgebra
using Arpack

function HOSVD(T, reqrank)
    X = hosvd(T, reqrank=reqrank)
    g = X.cten
    Us = X.fmat
    X = ttm(g, [Us...])
    return X
end

"""
input is real
"""
function STHOSVD(T, reqrank)
    N = ndims(T)
    tensor_shape = size(T)
    for i = 1 : N
        T_i = tenmat(T, i)
        if reqrank[i] == tensor_shape[i]
            USV = svd(T_i)
        else
            USV = svds(T_i; nsv=reqrank[i] )[1]
        end
        T = ttm( T, USV.U * USV.U', i)
    end
	return T
end
