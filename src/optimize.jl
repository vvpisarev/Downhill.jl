"""
    optimize!(M::Wrapper, fdf, x0)

Find an optimizer for `fdf`, starting with the initial approximation `x0`.
`fdf(x, g)` must return a tuple (f(x), ∇f(x)) and, if `g` is mutable, overwrite
it with the gradient.
"""
function optimize!(M::Wrapper, fdf, x0; reset=true, log_stream=nothing, verbosity=1)
    return __optim_impl!(M, fdf, x0, log_stream; reset, verbosity)
end

function __optim_impl!(M::Wrapper, fdf, x0, logger::AbstractLogger; reset=true)
    optfn! = OptFunc(M, fdf)

    with_logger(logger) do
        n = length(x0)
        @logmsg OptLogLevel """

        ==OPTIMIZATION START==
        # First $n value$(n == 1 ? "" : "s") - argument vector
        # next value - function value
        # last $n value$(n == 1 ? "" : "s") - gradient vector
        """
        @logmsg OptLogLevel "==SOLVER INITIALIZATION=="
        init!(M, optfn!, x0; reset)
        @logmsg OptLogLevel "==SOLVER INITIALIZED=="
        for niter in Iterators.countfrom(1)
            @logmsg OptLogLevel "# Iteration $niter"
            step!(M, optfn!)
            stopcond(M) && break
        end
        return convstat(M)
    end
end

function __optim_impl!(M::Wrapper, fdf, x0, log_stream::IO;
    reset=true, verbosity
)
    logger = ConsoleLogger(
        log_stream,
        LogLevel(-10 * verbosity);
        meta_formatter=metafmt_noprefix_nosuffix, show_limited=false,
    )
    __optim_impl!(M, fdf, x0, logger; reset)
end

function __optim_impl!(
    M::Wrapper, fdf, x0, log_stream::AbstractString;
    reset=true, verbosity::Integer
)
    open(log_stream, "w") do log_io
        logger = ConsoleLogger(
            log_io,
            LogLevel(-10 * verbosity);
            meta_formatter=metafmt_noprefix_nosuffix, show_limited=false,
        )
        return __optim_impl!(M, fdf, x0, logger; reset)
    end
end

function __optim_impl!(M::Wrapper, fdf, x0, ::Nothing; reset=true, kw...)
    __optim_impl!(M, fdf, x0, NullLogger(); reset)
end

"""
    optimize!(
        M::OptBuffer, fdf, x0;
        gtol=1e-6,
        maxiter=100,
        maxcalls=nothing,
        reset=true,
        constrain_step=nothing,
        log_stream=nothing,
        verbosity=1
    )

Find an optimizer for `fdf`, starting with the initial approximation `x0`.

# Arguments:
- `M::OptBuffer`: the core method to use for optimization
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
- `log_stream::Union{IO,AbstractString,Nothing}`: IO stream or a file name to log the
    optimization process or `nothing` to disable logging (default: `nothing`)
- `verbosity::Integer=1`: verbosity of logging. `1` (default) logs all points of objective
    function evaluation with corresponding values and gradients. `2` shows additional
    statistics regarding the line search.
"""
function optimize!(
    M::OptBuffer, fdf, x0;
    gtol = convert(float(eltype(x0)), 1e-6),
    maxiter = 100,
    maxcalls = nothing,
    reset = true,
    constrain_step = nothing,
    log_stream = nothing,
    verbosity::Integer=1,
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
    optimize!(M, fdf, x0; reset, log_stream, verbosity)
end

"""
    DescentMethods.solver(
        M::OptBuffer;
        gtol = 1e-6,
        maxiter = 100,
        maxcalls = nothing,
        constrain_step=nothing,
    )

Return the wrapper object for a chosen method to solve an optimization problem with given
    parameters. For the description of keywords, see [`optimize!`](@ref)
"""
function solver(
    M::OptBuffer;
    gtol=convert(eltype(argumentvec(M)), 1e-6),
    maxiter=100,
    maxcalls=nothing,
    constrain_step=nothing,
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
    return M
end

"""
    DescentMethods.solver(
        M::DataType, x;
        gtol=1e-6, maxiter = 100, maxcalls = nothing, constrain_step)

Return the wrapper object for a chosen method to solve an optimization problem with given
    parameters compatible with the dimensions of `x`.
"""
function solver(M::Type{<:OptBuffer}, x::AbstractVector; kw...)
    return solver(M(x); kw...)
end
