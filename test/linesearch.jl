"""
    convex_quadratic(Q, b)

Returns a convex quadratic function `fdf` with signature for `DescentMethods.strong_backtracking!`

    fdf(x, α, d) = f(x + α * d), ∇f(x + α * d)

where

      f(x) = 1/2 xᵀQx - bᵀx
     ∇f(x) = Qx - b
    ∇²f(x) = Q (hessian)

For f(x) to be convex, `Q` must be positive semi-definite matrix.
"""
function convex_quadratic(Q::AbstractMatrix, b::AbstractVector)
    @assert isposdef(Q)

    function f(x::AbstractVector, α::Number, d::AbstractVector)
        y = x + α * d
        y = convert(Vector{Float64}, y)
        val = 0.5 * y' * Q * y - b' * y
        grad = Q * y - b
        return (val, grad)
    end
    return f
end

function random_convex_quadratic(dim::Int)
    Q = rand(Float64, (dim, dim))
    Q = Q * Q'  # QQᵀ is always positive semi-definite
    b = rand(Float64, (dim, ))
    fdf = convex_quadratic(Q, b)
    return fdf, Q, b
end

"""
    convex_test(fdf, x0, Q[, d])

Perform linear search on a convex quadratic function `fdf` with hessian `Q`.
The search starts at `x0` with descent direction `d`.
Default `d` is opposite to gradient of `fdf`.

Return a tuple `(α, α₀)` where `α` is line search minimizer and `α₀` is analytical minimizer.

# Todo
1. Add randomness to default `d` vector.
"""
function convex_test(fdf, x0, Q, d=nothing)
    y0, grad0 = fdf(x0, 0, x0)  # f(x0 + α x0) = f(x0)
    if isnothing(d)
        d = - 0.01 * grad0  # for greater values of d ⟹ α ≉ α₀
    end

    α = DescentMethods.strong_backtracking!(fdf, x0, d, y0, grad0)

    "Exact minimizer of convex quadratic function. See Nocedal p. 56, eq. 3.55."
    α₀ = - (grad0' * d) / (d' * Q * d)

    return (α, α₀)
end

@testset "Convex quadratic" begin
    dim = 10
    fdf, Q, _ = random_convex_quadratic(dim)
    for i in 1:20
        x0 = rand(Float64, (dim, ))
        α, α₀ = convex_test(fdf, x0, Q)
        println(α, '\t', α₀)
        @test isapprox(α, α₀, rtol=0.05)
    end
end