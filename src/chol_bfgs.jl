export CholBFGS

function mcholesky!(A::AbstractMatrix{T}) where T
    δ = convert(T, 1e-3)
    β = convert(T, 1e3)
    n, m = size(A)
    m == n || throw(DimensionMismatch("Matrix is not square"))
    θ = zero(T)
    u = UpperTriangular(A)
    c = u
    @inbounds for j in 1:n
        c_jj = A[j,j]
        θ = zero(c_jj)
        for k in 1:j-1
            u[k,j] /= u[k,k]
        end
        for i in j+1:n
            c_ij = c[j,i]
            for k in 1:j-1
                c_ij -= u[k,j] * c[k,i] / u[k,k]
            end
            θ = max(abs(c_ij), θ)
            c[j,i] = c_ij
        end
        d_j = max(abs(c_jj), (θ / β)^2, δ)
        u[j,j] = sqrt(d_j)
        for i in j+1:n
            c[i,i] -= (c[j,i] / u[j,j])^2
        end
    end
    return A
end

"""
    CholBFGS
Quasi-Newton descent method.
"""
mutable struct CholBFGS{T<:AbstractFloat,
                        V<:AbstractVector{T},
                        C<:Cholesky{T}} <: CoreMethod
    hess::C
    x::V
    g::V
    xpre::V
    gpre::V
    d::V
    xdiff::V
    gdiff::V
    y::T
end

@inline gradientvec(M::CholBFGS) = M.g
@inline argumentvec(M::CholBFGS) = M.x
@inline step_origin(M::CholBFGS) = M.xpre


function CholBFGS(x::AbstractVector{T}) where {T}
    F = float(T)
    n = length(x)
    m = similar(x, F, (n, n))
    for j in 1:n, i in 1:n
        m[i,j] = (i == j)
    end
    cm = cholesky!(m)
    bfgs = CholBFGS(cm,
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                zero(F)
               )
    return bfgs
end

function init!(M::CholBFGS{T}, optfn!, x0; 
               reset, constrain_step = infstep) where {T}
    optfn!(x0, zero(T), x0)
    M.xpre .= x0
    M.xdiff .= abs.(x0) .+ 1
    if reset > 0
        M.xpre, M.x = M.x, M.xpre
        M.gpre, M.g = M.g, M.gpre
        map!(-, M.d, M.gpre)
        lmul!(reset, M.d)
        αmax = constrain_step(M.xpre, M.d)
        α = strong_backtracking!(optfn!, M.xpre, M.d, M.y, M.gpre, αmax = αmax, β = one(T)/100, σ = convert(T, 0.1))
        map!(-, M.xdiff, M.x, M.xpre)
        map!(-, M.gdiff, M.g, M.gpre)

        scale = dot(M.gdiff, M.gdiff) / dot(M.xdiff, M.gdiff)
        H = M.hess.factors
        nr, nc = size(H)
        for j in 1:nc, i in 1:nr
            H[i, j] = (i == j) * sqrt(abs(scale)) #sqrt(abs(M.gdiff[i] / M.xdiff[j]))
        end
    end
    return
end

@inline function reset!(M::CholBFGS)
    H = M.hess.factors
    nr, nc = size(H)
    for j in 1:nc, i in 1:nr
        H[i, j] = (i == j)
    end
    return
end

function reset!(M::CholBFGS, x0, scale::Real=1)
    copy!(M.x, x0)
    H = M.hess.factors
    nr, nc = size(H)
    for j in 1:nc, i in 1:nr
        # update consistent with BFGS
        H[i, j] = (i == j) / sqrt(scale)
    end
    return
end

function reset!(M::CholBFGS, x0, init_H::AbstractMatrix)
    copy!(M.x, x0)
    H = M.hess.factors
    copy!(H, init_H)
    mcholesky!(H)
    return
end

@inline function callfn!(M::CholBFGS, fdf, x, α, d)
    __update_arg!(M, x, α, d)
    y, g = fdf(M.x, M.g)
    __update_grad!(M, g)
    M.y = y
    return y, g
end

function __descent_dir!(M::CholBFGS)
    ldiv!(M.d, M.hess, M.gpre)
    lmul!(-1, M.d)
    return M.d
end

function step!(M::CholBFGS, optfn!; constrain_step = infstep)
    #=
    argument and gradient from the end of the last
    iteration are stored into `xpre` and `gpre`
    =#
    M.gpre, M.g = M.g, M.gpre
    M.xpre, M.x = M.x, M.xpre

    x, xpre, g, gpre, H = M.x, M.xpre, M.g, M.gpre, M.hess
    d = __descent_dir!(M)
    maxstep = constrain_step(xpre, d)
    α = strong_backtracking!(optfn!, xpre, d, M.y, gpre, αmax = maxstep, β = 0.01, σ = 0.9)
    if α > 0
        #=
        BFGS update:
                 Hδδ'H    γγ'
        H <- H - ------ + ---
                  δ'Hδ    δ'γ
        =#
        δ, γ = M.xdiff, M.gdiff
        γ .= g .- gpre
        δ .= x .- xpre

        U = UpperTriangular(H.factors)
        #=
        H = H.U' * H.U
        d <- H.U * δ
        δ'Hδ = d ⋅ d
        =#
        mul!(d, U, δ)
        d1 = dot(d, d)
        d2 = dot(δ, γ)
        # δ <- H.U' * H.U * δ = H * δ
        mul!(δ, U', d)
        rdiv!(δ, sqrt(d1))
        rdiv!(γ, sqrt(d2))
        lowrankupdate!(H, γ)
        lowrankdowndate!(H, δ)
    else
        fill!(M.xdiff, 0)
    end
    return α
end

@inline function __update_arg!(M::CholBFGS, x, α, d)
    map!(M.x, d, x) do a, b
        muladd(α, a, b)
    end
    return
end

@inline function __update_arg!(M::CholBFGS, x)
    if x !== M.x
        copy!(M.x, x)
    end
    return
end

@inline function __update_grad!(M::CholBFGS, g)
    if M.g !== g
        copy!(M.g, g)
    end
    return
end

function stopcond(M::CholBFGS{T}) where {T}
    rtol_x = 16 * eps(T)
    xdiff, xpre = M.xdiff, M.xpre
    for i in eachindex(xdiff, xpre)
        if abs(xdiff[i]) > rtol_x * abs(xpre[i])
            return false
        end
    end
    return true
end
