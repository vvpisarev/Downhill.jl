"""
    SteepestDescent

Descent method which minimizes the objective function in the direction
of antigradient at each step.
"""
mutable struct SteepestDescent{T<:AbstractFloat,V<:AbstractVector{T}} <: OptBuffer
    x::V
    xpre::V
    g::V
    dir::V
    y::T
    ypre::T
    α::T
end

function SteepestDescent(x::AbstractVector, α::Real)
    F = float(eltype(x))
    return SteepestDescent(
        similar(x, F),
        similar(x, F),
        similar(x, F),
        similar(x, F),
        F(NaN),
        F(NaN),
        convert(F, α),
    )
end

SteepestDescent(x::AbstractVector) = SteepestDescent(x, 1)

"""
`optfn!` must be the 3-arg closure that computes fdf(x + α*d) and overwrites `M`'s gradient
"""
function init!(optfn!, ::SteepestDescent{T}, x0; kw...) where {T}
    optfn!(x0, zero(T), x0)
    return
end

@inline reset!(::SteepestDescent) = nothing

function reset!(M::SteepestDescent, x0, α = M.α)
    if length(M.x) != length(x0)
        foreach((M.x, M.xpre, M.g, M.dir)) do v
            resize!(v, length(x0))
        end
    end
    M.α = α
    return
end

function callfn!(fdf, M::SteepestDescent, x, α, d)
    __update_arg!(M, x, α, d)
    y, g = fdf(M.x, M.g)
    __update_grad!(M, g)
    M.y = y
    return y, g
end

@inline function __descent_dir!(M::SteepestDescent)
    M.dir .= .-M.g
    return M.dir
end

function step!(optfn!, M::SteepestDescent; constrain_step = infstep)
    M.x, M.xpre = M.xpre, M.x
    M.ypre = M.y
    d = __descent_dir!(M)
    xpre = M.xpre
    αmax = constrain_step(xpre, d)
    α = strong_backtracking!(optfn!, xpre, d, M.ypre, M.g, α=M.α, αmax=αmax, β=1e-4, σ=0.1)
    return α
end

@inline function __update_arg!(M::SteepestDescent, x, α, d)
    map!(M.x, d, x) do a, b
        muladd(α, a, b)
    end
    return
end

@inline function __update_arg!(M::SteepestDescent, x)
    if x !== M.x
        copy!(M.x, x)
    end
    return
end

@inline function __update_xpre!(M::SteepestDescent, x)
    if x !== M.xpre
        copy!(M.xpre, x)
    end
    return
end

@inline function __update_grad!(M::SteepestDescent, g)
    if M.g !== g
        copy!(M.g, g)
    end
    return
end
