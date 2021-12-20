# Downhill.jl

```@meta
CurrentModule = Downhill
```

A collection of descent-based optimization methods.

The package is meant to be used for small-scale optimization problems. 
The use case is the problems where an optimization is some intermediate step 
that has to be run repeatedly.

## Basic usage

```julia
julia> function rosenbrock!(x::AbstractVector, g::AbstractVector; b=100)
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

julia> let x0 = zeros(2)
           opt = BFGS(x0)
           optresult = optimize!(rosenbrock!, opt, x0)
           optresult.argument
       end
2-element Vector{Float64}:
 0.9999999998907124
 0.9999999998080589
```