export FixedRateDescent
"""
    FixedRateDescent

Descent method which minimizes the objective function in the direction 
of antigradient at each step.
"""
mutable struct FixedRateDescent{T<:AbstractFloat,V<:AbstractVector{T}} <: DescentMethod
    x::V
    g::V
    α::T
end

@inline gradientvec(M::FixedRateDescent) = M.g
@inline argumentvec(M::FixedRateDescent) = M.x

function FixedRateDescent(x::AbstractVector, α::Real)
    FixedRateDescent(similar(x), similar(x), convert(eltype(x), α))
end

FixedRateDescent(x::AbstractVector) = FixedRateDescent(x, convert(eltype(x), 0.01))

function init!(M::FixedRateDescent{T}, optfn!, x0) where {T}
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

@inline function callfn!(M::FixedRateDescent, fdf, x, α, d)
    __update_arg!(M, x, α, d)
    y, g = fdf(M.x, M.g)
    __update_grad!(M, g)
    return y, g
end

function step!(M::FixedRateDescent, fdf!)
    fdf!(M.x, -M.α, M.g)
    return M.α
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

@inline isconverged(M::FixedRateDescent, gtol) = M |> gradientvec |> norm <= abs(gtol)


"""
    FixedRateDescent

Descent method which minimizes the objective function in the direction 
of antigradient at each step.
"""
mutable struct MomentumDescent{T<:AbstractFloat,V<:AbstractVector{T}} <: DescentMethod
    x::V
    g::V
    v::V
    learn_rate::T
    decay_rate::T
end

@inline gradientvec(M::MomentumDescent) = M.g
@inline argumentvec(M::MomentumDescent) = M.x

function MomentumDescent(x::AbstractVector; learn_rate::Real, decay_rate::Real)
    MomentumDescent(similar(x), similar(x), similar(x), convert(eltype(x), learn_rate), convert(eltype(x), decay_rate))
end

function init!(M::MomentumDescent{T}, optfn!, x0) where {T}
    optfn!(x0, zero(T), x0)
    fill!(M.v, zero(T))
    return
end

@inline function reset!(M::MomentumDescent)
    fill!(M.v, 0)
    return
end

function reset!(M::MomentumDescent, x0, learn_rate = M.learn_rate, decay_rate = M.decay_rate)
    if length(M.x) != length(x0)
        foreach((M.x, M.g, M.v)) do v
            resize!(v, length(x0))
        end
    end
    M.learn_rate = learn_rate
    M.decay_rate = decay_rate
    return
end

@inline function callfn!(M::MomentumDescent, fdf, x, α, d)
    __update_arg!(M, x, α, d)
    y, g = fdf(M.x, M.g)
    __update_grad!(M, g)
    return y, g
end

function step!(M::MomentumDescent, optfn!)
    map!(M.v, M.v, M.g) do v, g
        M.decay_rate * v - M.learn_rate * g
    end
    optfn!(M.x, 1.0, M.v)
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

@inline isconverged(M::MomentumDescent, gtol) = M |> gradientvec |> norm <= abs(gtol)