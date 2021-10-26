struct OptFunc{M<:DescentMethod,F<:Base.Callable}<:Function
    method::M
    fdf::F
end

(optfn::OptFunc)(x, α, d) = callfn!(optfn.method, optfn.fdf, x, α, d)

"""
    optimize!(M::Wrapper, fdf, x0)

Find an optimizer for `fdf`, starting with the initial approximation `x0`.
`fdf(x, g)` must return a tuple (f(x), ∇f(x)) and, if `g` is mutable, overwrite
it with the gradient.
"""
function optimize!(M::Wrapper, fdf, x0; reset=true, log_stream=nothing)
    return __optim_impl!(M, fdf, x0, log_stream; reset)
end

function __optim_impl!(M::Wrapper, fdf, x0, log_stream::IO; reset=true)
    logged_opt = Logger(M, log_stream)
    optfn! = OptFunc(logged_opt, fdf)

    init!(logged_opt, optfn!, x0; reset=reset)
    while !stopcond(logged_opt)
        step!(logged_opt, optfn!)
    end
    return convstat(logged_opt)
end

function __optim_impl!(M::Wrapper, fdf, x0, log_stream::AbstractString; reset=true)
    open(log_stream, "w") do log_io
        return __optim_impl!(M, fdf, x0, log_io; reset)
    end
end

function __optim_impl!(M::Wrapper, fdf, x0, ::Nothing; reset=true)
    optfn! = OptFunc(M, fdf)

    init!(M, optfn!, x0; reset=reset)
    while !stopcond(M)
        step!(M, optfn!)
    end
    return convstat(M)
end

function __optim_impl!(M::Logger, fdf, x0, ::Nothing; reset=true)
    optfn! = OptFunc(M, fdf)

    init!(M, optfn!, x0; reset=reset)
    while !stopcond(M)
        step!(M, optfn!)
    end
    return convstat(M)
end

function __optim_impl!(M::Logger, fdf, x0, log_stream; reset)
    throw(ArgumentError("Cannot wrap a logger into another logger"))
end
"""
    optimize!(
        M::CoreMethod, fdf, x0;
        gtol=1e-6,
        maxiter=100,
        maxcalls=nothing,
        reset=true,
        constrain_step=nothing,
        log_stream=nothing
    )

Find an optimizer for `fdf`, starting with the initial approximation `x0`.

# Arguments:
- `M::CoreMethod`: the core method to use for optimization
- `fdf(x, g)::Function`: function to optimize. It must return a tuple (f(x), ∇f(x)) and,
    if `g` is mutable, overwrite
    it with the gradient.
- `x0`: initial approximation

# Keywords:
- `gtol::Real`: stop optimization when the gradient norm is less
- `maxiter::Integer`: force stop optimization after this number of iterations
    (use `nothing` or a negative value to not constrain iteration number)
- `maxcalls::Integer`: force stop optimization after this number of function calls
    (use `nothing` or a negative value to not constrain call number)
- `reset=true`: a value to pass as a keyword argument to the optimizer `init!` method
- `constrain_step(x0, d)`: a function to constrain step from `x0` in the direction `d`.
    It must return a real-numbered value `α` such that `x0 + αd` is the maximum allowed step
- `log_stream::Union{IO,AbstractString}`: IO stream or a file name to log the optimization
    process
"""
function optimize!(
    M::CoreMethod, fdf, x0;
    gtol = convert(float(eltype(x0)), 1e-6),
    maxiter = 100,
    maxcalls = nothing,
    reset = true,
    constrain_step = nothing,
    log_stream = nothing
)
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
    optimize!(M, fdf, x0; reset=reset, log_stream=log_stream)
end

"""
    DescentMethods.solver(
        M::CoreMethod;
        gtol = 1e-6,
        maxiter = 100,
        maxcalls = nothing,
        constrain_step=nothing,
        log_stream=nothing
    )

Return the wrapper object for a chosen method to solve an optimization problem with given
    parameters. For the description of keywords, see [`optimize!`](@ref)
"""
function solver(
    M::CoreMethod;
    gtol=convert(eltype(argumentvec(M)), 1e-6),
    maxiter=100,
    maxcalls=nothing,
    constrain_step=nothing,
    log_stream=nothing
)
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
    if !isnothing(log_stream)
        M = Logger(M, log_stream)
    end
    return M
end

"""
    DescentMethods.solver(M::DataType, x; gtol = convert(eltype(x0), 1e-6), maxiter = 100, maxcalls = nothing, constrain_step)

Return the wrapper object for a chosen method to solve an optimization problem with given
    parameters compatible with the dimensions of `x`.
"""
function solver(M::Type{<:CoreMethod}, x::AbstractVector; kw...)
    return solver(M(x); kw...)
end
