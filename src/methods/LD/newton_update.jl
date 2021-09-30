using IterTools
using LinearAlgebra

function newton_update_theta(eta, eta_beta, theta)
    tensor_shape = size(eta)
    d = ndims(eta)
    idxs = []

    vecs_theta = []
    vecs_eta = []
    vecs_eta_beta = []
    for k = 1:d
        for i = 2:tensor_shape[k]
            idx = ntuple( j->j == k ? i : 1, d)
            push!(idxs, idx)
            push!(vecs_theta, theta[idx...])
            push!(vecs_eta, eta[idx...])
            push!(vecs_eta_beta, eta_beta[idx...])
        end
    end

    n_idxs = length(idxs)
    G = zeros(n_idxs,n_idxs)
    for gi = 1:n_idxs
        for gj = 1:n_idxs
            ix = ntuple( j->max(idxs[gi][j], idxs[gj][j]), d)
            G[gi,gj] += eta[ix...]
            G[gi,gj] -= eta[ idxs[gi]... ] * eta[ idxs[gj]...]
        end
    end

    # newton_update
    inv_G = inv(G)
    vecs_theta = vecs_theta - inv_G * (vecs_eta - vecs_eta_beta)

    for gi = 1:n_idxs
        theta[ idxs[gi]... ] = vecs_theta[gi]
    end

    return theta
end
