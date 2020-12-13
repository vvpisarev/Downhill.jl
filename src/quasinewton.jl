"""
    BFGS

Quasi-Newton descent method.
"""
mutable struct BFGS{T<:AbstractFloat,
                    V<:AbstractVector{T},
                    M<:AbstractMatrix{T}} <: CoreMethod
    invH::M
    x::V
    g::V
    xpre::V
    gpre::V
    d::V
    xdiff::V
    gdiff::V
    y::T
end

@inline gradientvec(M::BFGS) = M.g
@inline argumentvec(M::BFGS) = M.x
@inline step_origin(M::BFGS) = M.xpre

function BFGS(x::AbstractVector{T}) where {T}
    F = float(T)
    bfgs = BFGS(similar(x, F, (length(x), length(x))),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                zero(T)
               )
    reset!(bfgs)
    return bfgs
end

function init!(M::BFGS{T}, optfn!, x0) where {T}
    optfn!(x0, zero(T), x0)
    copy!(M.xpre, M.x)
    copy!(M.gpre, M.g)
    map!(-, M.d, M.g)
    α = strong_backtracking!(optfn!, M.xpre, M.d, M.y, M.gpre, α = 1e-4, β = 0.01, σ = 0.9)
    M.xdiff .= M.x - M.xpre
    M.gdiff .= M.g - M.gpre
    scale = dot(M.gdiff, M.xdiff) / dot(M.gdiff, M.gdiff)

    invH = M.invH
    nr, nc = size(invH)
    for j in 1:nc, i in 1:nr
        invH[i, j] = (i == j) * abs(M.xdiff[i] * M.gdiff[j])
    end
    return
end

@inline function reset!(M::BFGS)
    invH = M.invH
    nr, nc = size(invH)
    for j in 1:nc, i in 1:nr
        invH[i, j] = i == j
    end
    return
end

function reset!(M::BFGS, x0, scale::Real=1)
    copy!(M.x, x0)
    invH = M.invH
    nr, nc = size(invH)
    for j in 1:nc, i in 1:nr
        invH[i, j] = (i == j) * scale
    end
    return
end

@inline function callfn!(M::BFGS, fdf, x, α, d)
    __update_arg!(M, x, α, d)
    y, g = fdf(M.x, M.g)
    __update_grad!(M, g)
    M.y = y
    return y, g
end

function __descent_dir!(M::BFGS)
    mul!(M.d, M.invH, M.gpre, -1, 0)
    return M.d
end

@inline function __step_init!(M::BFGS, optfn!)
    #=
    argument and gradient from the end of the last
    iteration are stored into `xpre` and `gpre`
    =#
    M.gpre, M.g = M.g, M.gpre
    M.xpre, M.x = M.x, M.xpre
    return
end

function __compute_step!(M::BFGS, optfn!, d, maxstep)
    xpre = M.xpre
    α = strong_backtracking!(optfn!, xpre, d, M.y, M.gpre, αmax = maxstep, β = 0.01, σ = 0.9)
    #=
    BFGS update:
             δγ'B + Bγδ'   ⌈    γ'Bγ ⌉ δδ'
    B <- B - ----------- + |1 + -----| ---
                 δ'γ       ⌊     δ'γ ⌋ δ'γ
    =#
    δ, γ = M.xdiff, M.gdiff
    γ .= M.g .- M.gpre
    δ .= M.x .- M.xpre
    denom = dot(δ, γ)
    δscale = 1 + dot(γ, M.invH, γ) / denom
    # d <- B * γ
    mul!(M.d, M.invH, γ, 1, 0)
    M.invH .= M.invH .- (δ .* M.d' .+ M.d .* δ') ./ denom .+ δscale .* δ .* δ' ./ denom
    return α
end

@inline function __update_arg!(M::BFGS, x, α, d)
    map!(M.x, d, x) do a, b
        muladd(α, a, b)
    end
    return
end

@inline function __update_arg!(M::BFGS, x)
    if x !== M.x
        copy!(M.x, x)
    end
    return
end

@inline function __update_grad!(M::BFGS, g)
    if M.g !== g
        copy!(M.g, g)
    end
    return
end