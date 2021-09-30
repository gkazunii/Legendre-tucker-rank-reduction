using LinearAlgebra
using TensorToolbox

include("../utils/utils.jl")

function cost_function(cost, X, Xr)
    if cost == "LS"
        return norm(X-Xr)

    elseif cost == "KL"
        dkl = get_dkl(X, Xr)
        #return sum( X .* log.( X ./ Xr ) ) - sum(X) + sum(Xr)
        return dkl
    end
end

function reconst(S, A)
    N = ndims(S)
    Xr = S
    for n=1:N
        Xr = ttm(Xr, A[n], n)
    end
    return Xr
end

function init_based_SVD(X, reqrank)
    D = ndims(X)
    hosvd_X = hosvd(X, reqrank=reqrank)
    S = abs.(hosvd_X.cten)
    f = hosvd_X.fmat
    A = []
    for i = 1:D
        push!(A, abs.(f[i]))
    end

    return S, A
end

function init_random(tensor_size, reqrank)
    A = []
    N = length(reqrank)
    for n = 1:N
        An = rand(tensor_size[n], reqrank[n])
        push!(A, An)
    end
    S = rand( reqrank... )
    return S, A
end

"""
NMF Initializer. See Table 4 in the paper.
"""
function init_for_NMF(X, R; eps=1.0e-12, pre_iter_max_inloop=40)
    (m, l) = size(X)
    A = rand(m, R)
    S = rand(R, l)

    for iter = 1:pre_iter_max_inloop
        A .= max.(X*S', eps)
        A .= A .* ((X*S') ./ (A*S*S'))
        S .= max.(A'*X, eps)
        S .= S .* ( (A'*X) ./ (A'*A*S) )
    end
    return S, A
end

"""
Initializer based on NMF. See Table 5 in the paper.
"""
function init_based_NMF(X, reqrank; eps=1.0e-12, pre_iter_max=40, pre_iter_max_inloop=40)
    N = ndims(X)
    A = []
    for n=1:N
        _, An = init_for_NMF(tenmat(X, n), reqrank[n], pre_iter_max_inloop=pre_iter_max_inloop)
        push!(A, An)
    end

    S = rand( reqrank... )
    for iter = 1:pre_iter_max
        for n=1:N
            SAn = S
            for m in [1:n-1; n+1:N]
                SAn = ttm(SAn, A[m], m)
            end
            SAn = tenmat(SAn, n)
            A[n] = A[n] .* ( ( tenmat(X,n) * SAn' ) ./ (A[n] * SAn * SAn' ))
        end
        S = X
        for m=1:N
            S = ttm(S, A[m]', m)
        end
        S .= max.(S, eps)

        numerator = X
        denominator = S
        for m=1:N
            numerator = ttm(numerator, A[m]', m)
            denominator = ttm(denominator, A[m]'*A[m], m)
        end
        S .= S .* ( numerator ./ denominator )
    end
    return S, A
end

function update_An_LS(A, n, SAn, X)
    An = A[n] .* ( ( tenmat(X,n) * SAn' ) ./ (A[n] * SAn * SAn' ))
   return An
end

function update_An_KL(A, n, SAn, X)
    tensor_size = size(X)
    z = sum(SAn, dims=2)
    An = A[n] .* ((( tenmat(X,n) ./ ( A[n] * SAn ) ) * SAn' ) ./ ( ones(tensor_size[n]) * z' ))
    return An
end

function update_S_LS(S, A, X)
    N = ndims(S)
    numerator = X
    denominator = S
    for m=1:N
        numerator = ttm(numerator, A[m]', m)
        denominator = ttm(denominator, A[m]'*A[m], m)
    end
    S = S .* ( numerator ./ denominator )
    return S
end

function update_S_KL(S, A, X)
    N = ndims(S)
    numerator = X ./ reconst(S, A)
    denominator = ones( size(X)... )
    for m=1:N
        numerator = ttm(numerator, A[m]', m)
        denominator = ttm(denominator, A[m]', m)
    end
    S = S .* ( numerator ./ denominator )
    return S
end

function update(X, S, A, cost, max_iter=100, verbose=true, verbose_interval=20, tol=1.0e-5)
    N = ndims(S)

    cnt_iter = 0
    prev_LSerror = 0.0
    while(cnt_iter < max_iter)
        ############
        # update A #
        ############
        for n=1:N
            # get SAn
            SAn = S
            for m in [1:n-1;n+1:N]
                SAn = ttm(SAn, A[m], m)
            end
            SAn = tenmat(SAn, n)
            if cost == "LS"
                A[n] .= update_An_LS(A, n, SAn, X)
            elseif cost == "KL"
                A[n] .= update_An_KL(A, n, SAn, X)
            end
        end

        ########################
        # update Core tensor S #
        ########################
        if cost == "LS"
            S .= update_S_LS(S, A, X)
        elseif cost == "KL"
            S .= update_S_KL(S, A, X)
        end

        Xr = reconst(S, A)
        if verbose && (cnt_iter % verbose_interval == 0)
            error = cost_function(cost, X, Xr)
            println("$cnt_iter $error")
        end

        LSerror = cost_function("LS", X, Xr)
        if cnt_iter > 1 && abs(LSerror - prev_LSerror) < tol
            break
        end
        prev_LSerror = LSerror
        cnt_iter += 1
    end
    return S, A
end

"""
Non-Negative Tucker Decomposition
proposed by Young-Deok Kim et al.
See [original paper](https://ieeexplore.ieee.org/document/4270403)
[Implemented in also python](
http://tensorly.org/stable/_modules/tensorly/decomposition/_tucker.html#non_negative_tucker)

# Aruguments
- `X` : input non-negative tensor
- `reqrank` : Target Tucker rank, array
- `init_method` : initial values, "NMF" or "random"
- `cost` : cost function, "LS" or "KL"
- `verbose` : true or false
- `pre_iter_max` : iter_max of initialization based on "NMF"
- `pre_iter_max_inloop` : iter_max of initialization of "NMF"

The default values of `tol` and `max_iter` are decided by Tensorly implementation.
"""
function NTD(X, reqrank ;
        cost="LS", init_method="SVD", max_iter=10, verbose=true, verbose_interval=50,
        pre_iter_max=40, pre_iter_max_inloop=40, tol=1.0e-5)

    @assert ndims(X) == length(reqrank)
    tensor_size = size(X)

    #A[n] \in R^( tensor_size[n] \times reqrank[n] )
    #S \in R^(reqrank[1] \times ... \times reqrank[N])
    if init_method == "random"
        S, A = init_random(tensor_size, reqrank)
    elseif init_method == "NMF"
        S, A = init_based_NMF(X, reqrank, pre_iter_max=pre_iter_max, pre_iter_max_inloop=pre_iter_max_inloop)
    elseif init_method == "SVD"
        S, A = init_based_SVD(X, reqrank)
    else
        error("no init method ", init_method)
    end

    S, A = update(X, S, A, cost, max_iter, verbose, verbose_interval, tol)
    Xr = reconst(S, A)
    return S, A, Xr
end
