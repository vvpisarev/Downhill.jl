"""
    HyperGradDescent

Descent method which minimizes the objective function in the direction
of antigradient at each step.
"""
mutable struct HyperGradDescent{T<:AbstractFloat,V<:AbstractVector{T}} <: OptBuffer
    x::V
    xpre::V
    g::V
    gpre::V
    y::T
    ypre::T
    α::T
    μ::T
end

function HyperGradDescent(x::AbstractVector{T}, α::Real, μ::Real) where {T}
    F = float(T)
    return HyperGradDescent(
        similar(x, F),
        similar(x, F),
        similar(x, F),
        similar(x, F),
        F(NaN),
        F(NaN),
        convert(F, α),
        convert(F, μ),
    )
end

HyperGradDescent(x::AbstractVector{T}) where {T} = HyperGradDescent(x, 0, 1e-4)

function init!(optfn!, M::HyperGradDescent{T}, x0; reset, kw...) where {T}
    reset != false && reset!(M, x0)
    optfn!(x0, zero(T), x0)
    fill!(M.gpre, false)
    return M
end

@inline function reset!(M::HyperGradDescent)
    M.α = false
    return M
end

function reset!(M::HyperGradDescent{T}, x0, α = zero(T), μ = M.μ) where {T}
    if length(M.x) != length(x0)
        foreach((M.x, M.g, M.gpre)) do v
            resize!(v, length(x0))
        end
    end
    M.α = α
    M.μ = μ
    return M
end

@inline function callfn!(fdf, M::HyperGradDescent, x, α, d)
    __update_arg!(M, x, α, d)
    y, g = fdf(M.x, M.g)
    __update_grad!(M, g)
    return y, g
end

function step!(optfn!, M::HyperGradDescent; constrain_step = infstep)
    M.gpre, M.g = M.g, M.gpre
    M.α += abs(M.μ * dot(M.g, M.gpre))
    d = rmul!(M.gpre, -1)
    maxstep = constrain_step(M.x, d)
    s = maxstep > M.α ? M.α : maxstep / 2
    optfn!(M.x, s, d)
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
