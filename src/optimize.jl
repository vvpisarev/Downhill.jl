export optimize!

"""
    optimize!(M::DescentMethod, fdf, x0; gtol = 1e-6)

Find an optimizer for `fdf`, starting with the initial approximation `x0`. 
`fdf(x, g)` must return a tuple (f(x), ∇f(x)) and, if `g` is mutable, overwrite 
it with the gradient.
"""
function optimize!(M::DescentMethod, fdf, x0; gtol = convert(eltype(x0), 1e-6), maxsteps = 100)
    optfn!(x, α, d) = callfn!(M, fdf, x, α, d)

    init!(M, optfn!, x0)
    for _ in 1:maxsteps
        step!(M, optfn!)
        if isconverged(M, gtol)
            return argumentvec(M)
        end
    end
    g = gradientvec(M)
    @warn "Required tolerance not reached, gradient magnitude $(norm(g))"
    return argumentvec(M)
end