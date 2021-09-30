using TensorToolbox
using Random
using LinearAlgebra
Random.seed!(3)
include("LD.jl")

function main()
    T = rand( 100,100, 100)
    R = TLD(T)

    T = rand( 100,100, 100)
    R_newton = TLD_newton(T)

    println(" reconst_error by grad decent ",  norm(T - R) )
    println(" reconst_error by newton method ", norm(T - R_newton) )
end

main()
