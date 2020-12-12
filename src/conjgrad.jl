export CGDescent

"""
    CGDescent

Descent method which minimizes the objective function in the direction 
of antigradient at each step.
"""
mutable struct CGDescent{T<:AbstractFloat,V<:AbstractVector{T}} <: DescentMethod
    x::V
    xpre::V
    g::V
    gpre::V
    dir::V
    y::T
end

@inline gradientvec(M::CGDescent) = M.g
@inline argumentvec(M::CGDescent) = M.x

function CGDescent(x::AbstractVector)
    CGDescent(similar(x), similar(x), similar(x), similar(x), similar(x), zero(eltype(x)))
end

function __descent_dir!(M::CGDescent)
    β = dot(M.g, M.g) / dot(M.gpre, M.gpre)
    map!(M.dir, M.dir, M.g) do a, b
        muladd(a, β, -b)
    end
    return M.dir
end

function init!(M::CGDescent{T}, optfn!, x0) where {T}
    y, g = optfn!(x0, zero(T), x0)
    __update_gpre!(M, M.g)
    __zero_dir!(M)
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

function step!(M::CGDescent, fdf!)
    M.x, M.xpre = M.xpre, M.x
    xpre, d = M.xpre, __descent_dir!(M)
    M.g, M.gpre = M.gpre, M.g
    α = strong_backtracking!(fdf!, xpre, d, M.y, M.gpre, α = 0.01, β = 1e-4, σ = 0.1)
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

@inline isconverged(M::CGDescent, gtol) = M |> gradientvec |> norm <= abs(gtol)