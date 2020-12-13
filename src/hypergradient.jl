export HyperGradDescent

"""
    HyperGradDescent

Descent method which minimizes the objective function in the direction 
of antigradient at each step.
"""
mutable struct HyperGradDescent{T<:AbstractFloat,V<:AbstractVector{T}} <: CoreMethod
    x::V
    g::V
    gpre::V
    α::T
    μ::T
end

@inline gradientvec(M::HyperGradDescent) = M.g
@inline argumentvec(M::HyperGradDescent) = M.x

function HyperGradDescent(x::AbstractVector{T}, α::Real, μ::Real) where {T}
    F = float(T)
    HyperGradDescent(similar(x, F), similar(x, F), similar(x, F), convert(F, α), convert(F, μ))
end

HyperGradDescent(x::AbstractVector{T}) where {T} = HyperGradDescent(x, 0, 1e-4)

function init!(::HyperGradDescent{T}, optfn!, x0) where {T}
    optfn!(x0, zero(T), x0)
    return
end

@inline function reset!(M::HyperGradDescent)
    M.α = 0
end

function reset!(M::HyperGradDescent{T}, x0, α = zero(T), μ = M.μ) where {T}
    if length(M.x) != length(x0)
        foreach((M.x, M.g, M.gpre)) do v
            resize!(v, length(x0))
        end
    end
    M.α = α
    M.μ = μ
    return
end

@inline function callfn!(M::HyperGradDescent, fdf, x, α, d)
    __update_arg!(M, x, α, d)
    y, g = fdf(M.x, M.g)
    __update_grad!(M, g)
    return y, g
end

function step!(M::HyperGradDescent, optfn!)
    M.gpre, M.g = M.g, M.gpre
    M.α += M.μ * dot(M.g, M.gpre)
    optfn!(M.x, -M.α, M.gpre)
    return M.α
end

@inline function __update_arg!(M::HyperGradDescent, x, α, d)
    map!(M.x, d, x) do a, b
        muladd(α, a, b)
    end
    return
end

@inline function __update_arg!(M::HyperGradDescent, x)
    if x !== M.x
        copy!(M.x, x)
    end
    return
end

@inline function __update_grad!(M::HyperGradDescent, g)
    if M.g !== g
        copy!(M.g, g)
    end
    return
end

@inline isconverged(M::HyperGradDescent, gtol) = M |> gradientvec |> norm <= abs(gtol)