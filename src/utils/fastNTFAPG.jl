using TensorToolbox
using LinearAlgebra
using Arpack
using TensorDecompositions
using InvertedIndices
using Random
Random.seed!(123)
include("../methods.jl")

function get_tensor(A, reqrank, input_tensor_shape)
    N = length(input_tensor_shape)
    W = zeros(input_tensor_shape...)
    for r=1:reqrank
        Xr = ttt(A[1][:, r], A[2][:, r])
        for n=3:N
            Xr = ttt(Xr, A[n][:, r])
        end
        W = W + Xr
    end

    return W
end

function FastNTFAPG(Y, reqrank ; max_iter = 2, epsilon = 0.00001, tol = 0.0, verbose = true)
    # input is a nonnegative tensor
    # proposed by Guoxu Zhou in 2014
    # https://link.springer.com/content/pdf/10.1007/978-3-319-12436-0_51.pdf
    @assert length(Set(reqrank)) == 1 "Core Tensor should be cubic"
    reqrank = reqrank[1]

    N = ndims(Y)
    input_tensor_shape = size(Y)

    # any unconstrained CP decomposition algorithm
    init_pos = []
    for i=1:N
        push!(init_pos, randn(input_tensor_shape[i], reqrank))
    end
    # You can replace "ALS" by "SGSD".
    # The paper use CP-ALS. See implement detail in Sec4
    U = candecomp(Y, reqrank, Tuple(init_pos), verbose=false, method=Symbol("ALS"))

    for n = 1:N
        for r = 1 : reqrank
            U.factors[n][r] = U.factors[n][r] .* U.lambdas[r]
        end
    end
    U = U.factors

    # see ep.(13)
    A = []
    #for n = 1:N-1
    for n = 1:N
        # adjust sign
        for j = 1:reqrank
            if n != N
                i0 = argmax( abs.( U[n][:,j] ) )
                U[n][:, j] = U[n][:,j] .* sign(i0)
            else
                i0 = argmax( abs.( U[2][:,j] ) )
                U[N][:, j] = U[N][:,j] .* sign(i0)
            end
        end
        # initialiaze An
        push!(A, max.(U[n], epsilon)) #P+(Un) is max.(Un, 0)
    end

    iter = 1
    cost_at_init = norm( get_tensor(A, reqrank, input_tensor_shape) - Y)
    previous_cost = cost_at_init
    while iter <= max_iter
        for n = 1:N

            # Get G = Bn'Bn
            # See eq(11)
            ps = [1:N;][Not(n)]
            G = A[ps[1]]' * A[ps[1]]
            ps = ps[2:end]
            for p in ps
                G = G .* (A[p]'*A[p])
            end
            L = norm(G)

            # Get C=YnBn.
            # See ep(12)
            ps = [1:N;][Not(n)]
            UptAp = U[ps[1]]' * A[ps[1]]
            ps = ps[2:end]
            for p in ps
                UptAp = UptAp .* ( U[p]' * A[p] )
            end
            YnBn = U[n]*UptAp
            C = YnBn

            # sub lootuine in NTF_APG
            # see Algorithm 1. in paper
            k = 1
            previous_alphak = 1.0
            previous_Ak = A[n]
            Ak = A[n]
            Z = A[n]
            while k < 10
                Ak = max.(Z - 1.0 / L * ( C - previous_Ak*G), epsilon)
                alphak = (1.0 + sqrt( 4.0 * previous_alphak^2 + 1) ) / 2.0
                Z = Ak + (previous_alphak - 1.0) / alphak * (Ak - previous_Ak)

                k += 1
                previous_Ak = Ak
                previous_alphak = alphak
            end
            A[n] = Ak

            #if tol > 0 && iter % 1 == 0
            if true
                cost = norm( get_tensor(A, reqrank, input_tensor_shape) - Y)
                if verbose
                    println("iter: $iter cost: $cost")
                end
                #if (previous_cost - cost) / cost_at_init < tol
                #    break
                #end
                previous_cost = cost
            end

        end
        iter += 1

    end

    # reconstract
    G = zeros( fill(reqrank, N)... )
    for i = 1:reqrank
        idx = fill(i, N)
        G[idx...] = 1.0
    end

    X = ttm(G, A[1], 1)
    for n=2:N
        X = ttm(X, A[n], n)
    end

    return X
end

function main()
    Y = rand(6,6,6,6)
    X = FastNTFAPG(Y, [2,2,2,2], max_iter=1, epsilon = 1.0E-3)
    println( norm(Y-X) )

    X = lraSNTD(Y,[2,2,2,2])
    println( norm(Y-X) )

    X = NNCP(Y,[2,2,2,2])
    println( norm(Y-X) )
end

main()
