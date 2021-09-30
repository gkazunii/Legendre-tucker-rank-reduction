using TensorToolbox
using LinearAlgebra
using Arpack
#using TensorDecompositions

"""
hyper parameters max_iter and tol are decieded by sklearn
"""
function lranmf_mu(X, r ; max_iter=200, tol = 1.0E-3, verbose = false)
    # lranmf_mu is effective when r << min(n,m)
    # input is a onnegative matrix
    # proposed by Guoxu Zhou in 2012
    # https://ieeexplore.ieee.org/document/6166354

    n, m = size(X)
    epsilon = 0.000001

    # See step1 in section II-B in the paper
    if r == min(n, m)
        svd_X = svd(X)
    else
        svd_X = svds(X; nsv=r, ritzvec=true)[1]
    end
    Achil = svd_X.U * diagm(svd_X.S)
    Bchil = svd_X.V

    # See step2 in section II-B in the paper
    A = rand(n, r)
    B = rand(m, r)
    cost_at_init = norm(X - A*B')
    previous_cost = cost_at_init
    for iter = 1:max_iter
        B .= B .* ( max.( Bchil*(Achil' * A), epsilon ) ) ./ ( B*(A'*A) )
        A .= A .* ( max.( Achil*(Bchil' * B), epsilon ) ) ./ ( A*(B'*B) )

        if tol > 0 && iter % 10 == 0
            cost = norm(X - A*B')
            if verbose
                println("iter: $iter cost: $cost")
            end
            if (previous_cost - cost) / cost_at_init < tol
                break
            end
            previous_cost = cost
        end
    end

    # A * B' is rank-r matrix
    return A, B'
end

function lraSNTD(Y, reqrank)
    # Sequential nonnegative Tucker based on lraNMF
    # input is a nonnegative tensor
    # proposed by Guoxu Zhou in 2012
    # https://ieeexplore.ieee.org/document/6166354

    N = ndims(Y)
    input_tensor_shape = size(Y)

    # See section IV-2) in the paper
    # get non-negative factors
    A = []
    for n=1:N
        # Yn is Matricization of tensor Y by mode n
        Yn = tenmat(Y, n)
        An, _ = lranmf_mu(Yn, reqrank[n])
        push!(A, An)
        # ttm(Y,An,n) is mode-n product of tensor X and matrix An
        Y = ttm(Y, pinv(An), n)
    end

    # reproduce
    Y = ttm(Y, A[1], 1)
    for n=2:N
        Y = ttm(Y, A[n], n)
    end

    return Y
end
