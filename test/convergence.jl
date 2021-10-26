"""
    rosenbrock!(x, g; b=2)

Return the value of Rosenbrock function in `N` dimensions.
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

@testset "Convergence on the Rosenbrock function" begin
    dim = 2
    x0 = fill(-2, dim)
    @testset for method in OPT_TYPES
        opt = method(x0)
        optresult = optimize!(opt, rosenbrock!, x0; maxiter=1000)
        @test isapprox(optresult.argument, [1, 1], rtol=0.05)
    end
end
