"""
    BFGS

Quasi-Newton descent method.
"""
mutable struct BFGS{T<:AbstractFloat,
                    V<:AbstractVector{T},
                    M<:AbstractMatrix{T}} <: OptBuffer
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

@inline fnval(M::BFGS) = M.y
@inline gradientvec(M::BFGS) = M.g
@inline argumentvec(M::BFGS) = M.x
@inline step_origin(M::BFGS) = M.xpre

function BFGS(x::AbstractVector{T}) where {T}
    F = float(T)
    bfgs = BFGS(
        sqmatr(x, F),
        similar(x, F),
        similar(x, F),
        similar(x, F),
        similar(x, F),
        similar(x, F),
        similar(x, F),
        similar(x, F),
        zero(F)
    )
    reset!(bfgs)
    return bfgs
end

function init!(
    optfn!, M::BFGS{T}, x0;
    reset, constrain_step = infstep
) where {T}
    optfn!(x0, zero(T), x0)
    if reset
        M.xpre, M.x = M.x, M.xpre
        M.gpre, M.g = M.g, M.gpre
        map!(-, M.d, M.gpre)
        αmax = constrain_step(M.xpre, M.d)
        α = strong_backtracking!(
            optfn!, M.xpre, M.d, M.y, M.gpre;
            αmax=αmax, β=one(T)/100, σ=one(T)/10
        )
        map!(-, M.xdiff, M.x, M.xpre)
        map!(-, M.gdiff, M.g, M.gpre)

        invH = M.invH
        nr, nc = size(invH)
        scale = dot(M.xdiff, M.gdiff) / dot(M.gdiff, M.gdiff)
        fill!(invH, 0)
        for i in 1:nc
            elt = M.xdiff[i] / M.gdiff[i]
            invH[i, i] = 1e-5 < elt / scale < 1e5 ? elt : scale
        end
    end
    return M
end

@inline function reset!(M::BFGS)
    invH = M.invH
    nr, nc = size(invH)
    for j in 1:nc, i in 1:nr
        invH[i, j] = i == j
    end
    return M
end

function reset!(M::BFGS, x0, scale::Real=1)
    copy!(M.x, x0)
    invH = M.invH
    nr, nc = size(invH)
    for j in 1:nc, i in 1:nr
        invH[i, j] = (i == j) * scale
    end
    return M
end

@inline function callfn!(fdf, M::BFGS, x, α, d)
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

function step!(optfn!::F, M::BFGS; constrain_step::S=infstep) where {F,S}
    #=
    argument and gradient from the end of the last
    iteration are stored into `xpre` and `gpre`
    =#
    M.gpre, M.g = M.g, M.gpre
    M.xpre, M.x = M.x, M.xpre

    x, xpre, g, gpre, invH = M.x, M.xpre, M.g, M.gpre, M.invH
    d = __descent_dir!(M)
    maxstep = constrain_step(xpre, d)
    α = strong_backtracking!(optfn!, xpre, d, M.y, gpre, αmax = maxstep, β = 0.01, σ = 0.9)
    if α > 0
        #=
        BFGS update:
                δγ'B + Bγδ'   ⌈    γ'Bγ ⌉ δδ'
        B <- B - ---------- + |1 + -----| ---
                    δ'γ       ⌊     δ'γ ⌋ δ'γ
        =#
        δ, γ = M.xdiff, M.gdiff
        map!(-, γ, g, gpre)
        map!(-, δ, x, xpre)
        denom = dot(δ, γ)
        δscale = 1 + dot(γ, invH, γ) / denom
        # d <- B * γ
        mul!(d, invH, γ, 1, 0)
        invH .= invH .- (δ .* d' .+ d .* δ') ./ denom .+ δscale .* δ .* δ' ./ denom
    else
        fill!(M.xdiff, 0)
    end
    return α
end

@inline function __update_arg!(M::BFGS, x, α, d)
    map!(M.x, d, x) do a, b
        muladd(α, a, b)
    end
    return M.x
end

@inline function __update_arg!(M::BFGS, x)
    if x !== M.x
        copy!(M.x, x)
    end
    return M.x
end

@inline function __update_grad!(M::BFGS, g)
    if M.g !== g
        copy!(M.g, g)
    end
    return M.g
end

function stopcond(M::BFGS{T}) where {T}
    rtol_x = 16 * eps(T)
    xdiff, xpre = M.xdiff, M.xpre
    for i in eachindex(xdiff, xpre)
        if abs(xdiff[i]) > rtol_x * abs(xpre[i])
            return false
        end
    end
    return true
end
