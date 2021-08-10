"""
    SteepestDescent

Descent method which minimizes the objective function in the direction
of antigradient at each step.
"""
mutable struct SteepestDescent{T<:AbstractFloat,V<:AbstractVector{T}} <: CoreMethod
    x::V
    xpre::V
    g::V
    dir::V
    y::T
    α::T
end

@inline fnval(M::SteepestDescent) = M.y
@inline gradientvec(M::SteepestDescent) = M.g
@inline argumentvec(M::SteepestDescent) = M.x
@inline step_origin(M::SteepestDescent) = M.xpre

function SteepestDescent(x::AbstractVector, α::Real)
    F = float(eltype(x))
    SteepestDescent(similar(x, F), similar(x, F), similar(x, F), similar(x, F), zero(F), convert(F, α))
end

SteepestDescent(x::AbstractVector) = SteepestDescent(x, 1)

"""
`optfn!` must be the 3-arg closure that computes fdf(x + α*d) and overwrites `M`'s gradient
"""
function init!(::SteepestDescent{T}, optfn!, x0; kw...) where {T}
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

function callfn!(M::SteepestDescent, fdf, x, α, d)
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

function step!(M::SteepestDescent, optfn!; constrain_step = infstep)
    M.x, M.xpre = M.xpre, M.x
    d = __descent_dir!(M)
    xpre = M.xpre
    αmax = constrain_step(xpre, d)
    α = strong_backtracking!(optfn!, xpre, d, M.y, M.g, α=M.α, αmax=αmax, β=1e-4, σ=0.1)
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
