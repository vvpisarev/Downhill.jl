export optimize!

"""
    optimize!(M::DescentWrapper, fdf, x0)

Find an optimizer for `fdf`, starting with the initial approximation `x0`. 
`fdf(x, g)` must return a tuple (f(x), ∇f(x)) and, if `g` is mutable, overwrite 
it with the gradient.
"""
function optimize!(M::Wrapper, fdf, x0; reset = true)
    optfn!(x, α, d) = callfn!(M, fdf, x, α, d)

    init!(M, optfn!, x0, reset = reset)
    while !stopcond(M)
        step!(M, optfn!)
    end
    return convstat(M)
end

"""
    optimize!(M::CoreMethod, fdf, x0; gtol = 1e-6, maxiter = 100, maxcalls = nothing, reset = true, constrain_step = nothing)

Find an optimizer for `fdf`, starting with the initial approximation `x0`. 
`fdf(x, g)` must return a tuple (f(x), ∇f(x)) and, if `g` is mutable, overwrite 
it with the gradient. A function `constrain_step(x0, d)` may be passed to limit 
the step sizes.
"""
function optimize!(M::CoreMethod, fdf, x0;
                   gtol = convert(eltype(x0), 1e-6),
                   maxiter = 100,
                   maxcalls = nothing,
                   reset = true,
                   constrain_step = nothing,
                   track_io = nothing)
    if !isnothing(gtol) && gtol > 0
        M = StopByGradient(M, gtol)
    end
    if isnothing(maxiter) || maxiter < 0
        M = LimitIters(M)
    else
        M = LimitIters(M, maxiter)
    end
    if isnothing(maxcalls) || maxcalls < 0
        M = LimitCalls(M)
    else
        M = LimitCalls(M, maxcalls)
    end
    if !isnothing(constrain_step)
        M = ConstrainStepSize(M, constrain_step)
    end
    if track_io isa IO
        M = TrackPath(M, track_io)
    end
    optimize!(M, fdf, x0, reset = reset)
end

infstep(x0, d) = convert(eltype(d), Inf)