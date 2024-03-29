infstep(x0, d) = convert(eltype(d), Inf)
stopbygradient(tol::Real) = (x, xpre, y, ypre, g) -> norm(g, 2) ≤ tol

#=
A generic way to create a mutable N×N matrix from a length-N vector.
Behaves like `similar(vec)` but returns `MMatrix` if the
argument is a `StaticVector` (`similar(staticvec, (n,n))` creates a
`Base.Matrix` which is less performant, and I don't want an explicit
dependency on `StaticArrays.jl` - V.P.)
=#
"""
    sqmatr(vec::AbstractVector, [element_type = eltype(vec)])

Create an uninitialized mutable N×N matrix with the given element type, given a
vector of length N.
"""
function sqmatr(vec::AbstractVector{T}, F::DataType = T) where {T}
    sz = axes(vec)
    return similar(vec, F, (sz..., sz...))
end

"""
Return the maximum absolute values of diagonal and off-diagonal elements of `A`.
"""
function γξ(A::AbstractMatrix)
    n = LinearAlgebra.checksquare(A)
    γ = ξ = zero(eltype(A))
    for j in 1:n
        for i in 1:j-1
            ξ = max(ξ, abs(A[i,j]))
        end
        γ = max(γ, abs(A[j,j]))
    end
    return γ, ξ
end

"""
    mcholesky!(A::AbstractMatrix)

Perform an in-place modified Cholesky decomposition on matrix `A`
(Gill, Murray, Wright, Practical optimization (1981), p.111)
"""
function mcholesky!(A::AbstractMatrix{T}; δ=eps(T)^(2/3)) where {T}
    n = LinearAlgebra.checksquare(A)
    γ, ξ = γξ(A)
    ν = max(1, sqrt(n^2 - 1))
    β² = max(γ, ξ / ν, eps(T))
    u = UpperTriangular(A)
    δA = δ * γ # similar to the choice of Schnabel-Eskow (1999)
    @inbounds for j in 1:n
        c_jj = u[j,j]
        θ = zero(c_jj)
        for k in 1:j-1
            u[k,j] /= u[k,k]
        end
        for i in j+1:n
            c_ij = u[j,i]
            for k in 1:j-1
                c_ij -= u[k,j] * u[k,i] / u[k,k]
            end
            θ = max(abs(c_ij), θ)
            u[j,i] = c_ij
        end
        d_j = max(abs(c_jj), θ^2 / β², δA)
        u_jj = sqrt(d_j)
        u[j,j] = u_jj
        for i in j+1:n
            u[i,i] -= (u[j,i] / u_jj)^2
        end
    end
    return Cholesky(A, 'U', 0)
end


#####LOGGING#####

function metafmt_noprefix_nosuffix(level::LogLevel, _module, group, id, file, line)
    @nospecialize
    if level < Info
        color = default_logcolor(level)
        prefix = string(
            level == OptLogLevel ? "Optimizer" :
            level == LSLogLevel ? "Linesearch" :
            string(level),
            ':'
        )
        suffix::String = ""
        return color, prefix, suffix
    else
        return default_metafmt(level, _module, group, id, file, line)
    end
end
