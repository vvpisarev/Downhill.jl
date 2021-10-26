using Test

using DescentMethods
using DescentMethods: argumentvec, gradientvec, step_origin, mcholesky!

using  LinearAlgebra

using Random

const test_rng = MersenneTwister(123581321)

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
