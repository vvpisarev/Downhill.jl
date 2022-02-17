using Test

using Downhill
using Downhill: argumentvec, gradientvec, step_origin, fnval, fnval_origin
using Downhill: mcholesky!

using LinearAlgebra

using Random

const test_rng = MersenneTwister(123581321)

"""
    rosenbrock!(x, g; b=2)

Return the value of Rosenbrock function in `length(x)` dimensions.
"""
function rosenbrock!(x::AbstractVector, g::AbstractVector; b=2)
    f = zero(eltype(g))
    fill!(g, 0)
    inds = eachindex(x, g)
    for i in 2:last(inds)
        f += (1 - x[i-1])^2 + b * (x[i] - x[i-1]^2)^2
        g[i-1] += 2 * (x[i-1] - 1) + 4 * b * x[i-1] * (x[i-1]^2 - x[i])
        g[i] += 2 * b * (x[i] - x[i-1]^2)
    end
    return f, g
end

OPT_TYPES = (
    SteepestDescent,
    HyperGradDescent,
    FixedRateDescent,
    MomentumDescent,
    NesterovMomentum,
    CGDescent,
    BFGS,
    CholBFGS,
)

@testset "Testing package DescentMethods" begin
# Test that all methods have a constructor
include("constructors.jl")

# Test that all methods have interface functions
include("interface.jl")

# Test linesearch on quadratic functions
include("linesearch.jl")

# Test modified Cholesky decomposition
include("cholesky.jl")

# Test convergence on the Rosenbrock function
include("convergence.jl")
end
