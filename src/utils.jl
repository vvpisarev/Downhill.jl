#=
A generic way to create a mutable N×N matrix from a length-N vector.
Behaves like `similar(vec)` but returns `MMatrix` if the 
argument is a `StaticVector` (`similar(staticvec, (n,n))` creates a
`Base.Matrix` which is less performant, and I don't want an explicit
dependency on `StaticArrays.jl` - V.P.)
=#
function sqmatr(vec::AbstractVector{T}, F::DataType = T) where {T}
    cinds = CartesianIndices((eachindex(vec), eachindex(vec)))
    return similar(cinds, F)
end