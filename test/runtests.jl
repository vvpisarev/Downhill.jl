using Test, DescentMethods, LinearAlgebra

@testset "Testing package DescentMethods" begin
# Test that all methods have a constructor
include("constructors.jl")

# Test that all methods have interface functions
include("interface.jl")

# Test linesearch on quadratic functions
include("linesearch.jl")

end