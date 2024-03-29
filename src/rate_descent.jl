"""
    FixedRateDescent

Descent method which minimizes the objective function in the direction
of antigradient at each step.
"""
mutable struct FixedRateDescent{T<:AbstractFloat,V<:AbstractVector{T}} <: OptBuffer
    x::V
    xpre::V
    g::V
    α::T
    y::T
    ypre::T
end

function FixedRateDescent(x::AbstractVector, α::Real)
    F = float(eltype(x))
    return FixedRateDescent(
        similar(x, F),
        similar(x, F),
        similar(x, F),
        convert(F, α),
        F(NaN),
        F(NaN),
    )
end

FixedRateDescent(x::AbstractVector) = FixedRateDescent(x, 0.01)

function init!(optfn!, M::FixedRateDescent{T}, x0; kw...) where {T}
    optfn!(x0, zero(T), x0)
    return
end

@inline reset!(::FixedRateDescent) = nothing

function reset!(M::FixedRateDescent, x0, α = M.α)
    if length(M.x) != length(x0)
        foreach((M.x, M.g)) do v
            resize!(v, length(x0))
        end
    end
    M.α = α
    return
end

@inline function callfn!(fdf, M::FixedRateDescent, x, α, d)
    __update_arg!(M, x, α, d)
    y, g = fdf(M.x, M.g)
    __update_grad!(M, g)
    return y, g
end

function step!(optfn!, M::FixedRateDescent; constrain_step = infstep)
    d = rmul!(M.g, -1)
    maxstep = constrain_step(M.x, d)
    s = M.α <= maxstep ? M.α : maxstep / 2
    optfn!(M.x, s, d)
    return s
end

@inline function __update_arg!(M::FixedRateDescent, x, α, d)
    map!(M.x, d, x) do a, b
        muladd(α, a, b)
    end
    return
end

@inline function __update_arg!(M::FixedRateDescent, x)
    if x !== M.x
        copy!(M.x, x)
    end
    return
end

@inline function __update_grad!(M::FixedRateDescent, g)
    if M.g !== g
        copy!(M.g, g)
    end
    return
end

"""
    MomentumDescent

Descent method which minimizes the objective function in the direction
of antigradient at each step.
"""
mutable struct MomentumDescent{T<:AbstractFloat,V<:AbstractVector{T}} <: OptBuffer
    x::V
    xpre::V
    g::V
    v::V
    learn_rate::T
    decay_rate::T
    y::T
    ypre::T
end

function MomentumDescent(x::AbstractVector; learn_rate::Real=0.01, decay_rate::Real=0.9)
    F = float(eltype(x))
    return MomentumDescent(
        similar(x, F),
        similar(x, F),
        similar(x, F),
        similar(x, F),
        convert(F, learn_rate),
        convert(F, decay_rate),
        F(NaN),
        F(NaN),
    )
end

function init!(optfn!, M::MomentumDescent{T}, x0; kw...) where {T}
    optfn!(x0, zero(T), x0)
    fill!(M.v, zero(T))
    return M
end

@inline function reset!(M::MomentumDescent)
    fill!(M.v, false)
    return M
end

function reset!(
    M::MomentumDescent,
    x0,
    learn_rate = M.learn_rate,
    decay_rate = M.decay_rate,
)
    n = length(x0)
    if length(M.x) != n
        for v in (M.x, M.g, M.v)
            resize!(v, n)
        end
    end
    M.v .= false
    M.x .= x0
    M.learn_rate = learn_rate
    M.decay_rate = decay_rate
    return M
end

@inline function callfn!(fdf, M::MomentumDescent, x, α, d)
    __update_arg!(M, x, α, d)
    y, g = fdf(M.x, M.g)
    __update_grad!(M, g)
    return y, g
end

function step!(optfn!, M::MomentumDescent; constrain_step = infstep)
    map!(M.v, M.v, M.g) do v, g
        M.decay_rate * v - M.learn_rate * g
    end
    maxstep = constrain_step(M.x, M.v)
    if maxstep <= 1
        M.v *= maxstep / 2
    end
    optfn!(M.x, one(maxstep), M.v)
    return M.learn_rate
end

@inline function __update_arg!(M::MomentumDescent, x, α, d)
    map!(M.x, d, x) do a, b
        muladd(α, a, b)
    end
    return
end

@inline function __update_arg!(M::MomentumDescent, x)
    if x !== M.x
        copy!(M.x, x)
    end
    return
end

@inline function __update_grad!(M::MomentumDescent, g)
    if M.g !== g
        copy!(M.g, g)
    end
    return
end

"""
    NesterovMomentum

Descent method which minimizes the objective function in the direction
of antigradient at each step.
"""
mutable struct NesterovMomentum{T<:AbstractFloat,V<:AbstractVector{T}} <: OptBuffer
    x::V
    xpre::V
    g::V
    v::V
    α::T # learning rate
    β::T # decay rate
    y::T
    ypre::T
end

function NesterovMomentum(x::AbstractVector; learn_rate::Real=0.01, decay_rate::Real=0.9)
    F = float(eltype(x))

    # α and β are parameters of "half step" v <- βv - α∇f
    # applied twice, it must yield v <- decay_rate × v - learn_rate × ∇f
    α = learn_rate
    β = decay_rate
    return NesterovMomentum(
        similar(x, F),
        similar(x, F),
        similar(x, F),
        similar(x, F),
        convert(F, α),
        convert(F, β),
        F(NaN),
        F(NaN),
    )
end

function init!(optfn!, M::NesterovMomentum{T}, x0; kw...) where {T}
    optfn!(x0, zero(T), x0)
    fill!(M.v, zero(T))
    return M
end

@inline function reset!(M::NesterovMomentum)
    fill!(M.v, false)
    return M
end

function reset!(M::NesterovMomentum, x0, learn_rate=M.learn_rate, decay_rate=M.decay_rate)
    n = length(x0)
    if length(M.x) != n
        for v in (M.x, M.g, M.v)
            resize!(v, n)
        end
    end
    M.α = learn_rate / (1 + decay_rate)
    M.β = sqrt(decay_rate)
    return M
end

@inline function callfn!(fdf, M::NesterovMomentum, x, α, d)
    __update_arg!(M, x, α, d)
    y, g = fdf(M.x, M.g)
    __update_grad!(M, g)
    return y, g
end

function step!(optfn!, M::NesterovMomentum; constrain_step = infstep)
    α, β = M.α, M.β
    M.v *= β
    maxstep = constrain_step(M.x, M.v)
    if maxstep <= 1
        M.v *= maxstep / 2
    end
    optfn!(M.x, one(maxstep), M.v)
    M.v .-= α * M.g
    M.x .-= α * M.g
    return M.α
end

@inline function __update_arg!(M::NesterovMomentum, x, α, d)
    map!(M.x, d, x) do a, b
        muladd(α, a, b)
    end
    return
end

@inline function __update_arg!(M::NesterovMomentum, x)
    if x !== M.x
        copy!(M.x, x)
    end
    return
end

@inline function __update_grad!(M::NesterovMomentum, g)
    if M.g !== g
        copy!(M.g, g)
    end
    return
end
