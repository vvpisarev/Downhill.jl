using DescentMethods

"""
    rosenbrock(x; b=2)

Return the value of Rosenbrock function
    ```math
    \\sum_{i=1}^n \\right[(1 - x_{i-1})^2 + b (x_i - x_{i-1})^2 \\left]
    ```
    in `n = length(x)` dimensions.
"""
function rosenbrock(x::AbstractVector{T}; b=2) where {T}
    f = zero(T) * zero(b) # type-generic zero
    for i in firstindex(x)+1:lastindex(x)
        f += (one(T) - x[i-1])^2 + b * (x[i] - x[i-1]^2)^2
    end
    return f
end

"""
    drosenbrock(x; b=2)

Return the gradient of Rosenbrock function
    ```math
    \\sum_{i=1}^n \\right[(1 - x_{i-1})^2 + b (x_i - x_{i-1})^2 \\left]
    ```
    in `n = length(x)` dimensions.
"""
function drosenbrock(x::AbstractVector{T}; b=2) where {T}
    g = fill!(similar(x, promote_type(T, typeof(b))), 0)
    for i in firstindex(x)+1:lastindex(x)
        g[i-1] += 2 * (x[i-1] - 1) + 4 * b * x[i-1] * (x[i-1]^2 - x[i])
        g[i] += 2 * b * (x[i] - x[i-1]^2)
    end
    return g
end

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

ros(x, g) = (rosenbrock(x), drosenbrock(x))

ros!(x, g) = rosenbrock!(x, g)

ans_nonmutating = let x0 = zeros(2)
    opt = BFGS(x0)
    optresult = optimize!(opt, ros, x0; maxiter=1000, log_stream=tempname(cleanup=false), verbosity=2)
    println("""
        ==Nonmutating gradient evaluation==

        Optimization converged: $(optresult.converged)
        Final argument: $(optresult.argument)
        Final gradient: $(optresult.gradient)
        Iteration count: $(optresult.iterations)
        Call count: $(optresult.calls)
    """)
    optresult.argument
end

ans_mutating = let x0 = zeros(2)
    opt = BFGS(x0)
    optresult = optimize!(opt, ros!, x0; maxiter=1000)
    println("""
        ==Mutating gradient evaluation==

        Optimization converged: $(optresult.converged)
        Final argument: $(optresult.argument)
        Final gradient: $(optresult.gradient)
        Iteration count: $(optresult.iterations)
        Call count: $(optresult.calls)
    """)
    optresult.argument
end

@assert ans_nonmutating ≈ ans_mutating
