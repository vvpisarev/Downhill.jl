export CGDescent

"""
    CGDescent

Conjugate gradient method (Hager-Zhang version [W.Hager, H.Zhang // SIAM J. Optim (2006) Vol. 16, pp. 170-192]) 
"""
mutable struct CGDescent{T<:AbstractFloat,V<:AbstractVector{T}} <: CoreMethod
    x::V
    xpre::V
    g::V
    gpre::V
    gdiff::V
    dir::V
    y::T
    α::T
    α0::T
end

@inline fnval(M::CGDescent) = M.y
@inline gradientvec(M::CGDescent) = M.g
@inline argumentvec(M::CGDescent) = M.x
@inline step_origin(M::CGDescent) = M.xpre

function CGDescent(x::AbstractVector)
    T = eltype(x)
    CGDescent(similar(x),
              similar(x),
              similar(x),
              similar(x),
              similar(x),
              similar(x),
              zero(T),
              convert(T, 0.01),
              convert(T, 0.01)
             )
end

function __descent_dir!(M::CGDescent)
    d = M.dir
    y = M.gdiff
    dty = dot(d, y)
    if iszero(dty)
        # d' * y == 0 means we are at the first iteration
        β = dty
    else
        β = (dot(y, M.g) - 2 * dot(d, M.g) * dot(y, y) / dty) / dty
    end
    η = one(β) / 100
    β = max(β, -1 / (norm(d) * min(η, norm(M.g))))
    map!(d, d, M.g) do a, b
        muladd(a, β, -b)
    end
    return d
end

function init!(M::CGDescent{T}, optfn!, x0; reset, constrain_step = infstep) where {T}
    y, g = optfn!(x0, zero(T), x0)
    __update_gpre!(M, M.g)
    map!(-, M.dir, M.g)
    M.α = M.α0
    return
end

@inline reset!(::CGDescent) = nothing

function reset!(M::CGDescent, x0)
    if length(M.x) != length(x0)
        foreach((M.x, M.xpre, M.g, M.gpre, M.dir)) do v
            resize!(v, length(x0))
        end
    end
    return
end

function callfn!(M::CGDescent, fdf, x, α, d)
    __update_arg!(M, x, α, d)
    let x = argumentvec(M)
        (y, g) = fdf(x, gradientvec(M))
        __update_grad!(M, g)
        M.y = y
        return (y, g)
    end
end

function step!(M::CGDescent, optfn!; constrain_step = infstep)
    M.x, M.xpre = M.xpre, M.x
    map!(-, M.gdiff, M.g, M.gpre)

    d = __descent_dir!(M)
    xpre = M.xpre
    M.g, M.gpre = M.gpre, M.g
    ypre = M.y
    maxstep = constrain_step(xpre, d)
    α = strong_backtracking!(optfn!, xpre, d, ypre, M.gpre, α = M.α, αmax = maxstep, β = 0.01, σ = 0.1)
    fdiff = M.y - ypre
    if fdiff < 0
        M.α = 2 * fdiff / dot(d, M.gpre)
    end
    return α
end

function __update_arg!(M::CGDescent, x, α, d)
    map!(M.x, d, x) do a, b
        muladd(α, a, b)
    end
    return
end

@inline function __update_arg!(M::CGDescent, x)
    if x !== M.x
        copy!(M.x, x)
    end
    return
end

@inline function __update_xpre!(M::CGDescent, x)
    if x !== M.xpre
        copy!(M.xpre, x)
    end
    return
end

@inline function __update_grad!(M::CGDescent, g)
    if M.g !== g
        copy!(M.g, g)
    end
    return
end

@inline function __update_gpre!(M::CGDescent, g)
    if M.gpre !== g
        copy!(M.gpre, g)
    end
    return
end

@inline function __zero_dir!(M::CGDescent)
    fill!(M.dir, 0)
    return
end