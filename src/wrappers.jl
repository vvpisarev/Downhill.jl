function base_method(::AbstractOptBuffer) end

argumentvec(M::Wrapper) = argumentvec(base_method(M))
gradientvec(M::Wrapper) = gradientvec(base_method(M))
step_origin(M::Wrapper) = step_origin(base_method(M))
fnval(M::Wrapper) = fnval(base_method(M))
fnval_origin(M::Wrapper) = fnval_origin(base_method(M))

__descent_dir!(M::Wrapper) = __descent_dir!(base_method(M))

@inline callfn!(fdf, M::Wrapper, x, α, d) = callfn!(fdf, base_method(M), x, α, d)

function init!(fdf, M::Wrapper, args...; kw...)
    init!(fdf, base_method(M), args...; kw...)
    return
end

"""
    reset!(M::AbstractOptBuffer, args...; kwargs...)

Reset the solver parameters to the default (or to specific value -- see the documentation
    for the specific types).

Each method has to implement a parameter-free `reset!(M)` method.
"""
function reset!(M::Wrapper, args...; kw...)
    reset!(base_method(M), args...; kw...)
    return
end

step!(fn::F, M::Wrapper; kw...) where {F} = step!(fn, base_method(M); kw...)

stopcond(M::Wrapper) = stopcond(base_method(M))
@inline stopcond(M::OptBuffer) = false

conv_success(M::Wrapper) = conv_success(base_method(M))
@inline conv_success(M::OptBuffer) = false

iter_count(M::Wrapper) = iter_count(base_method(M))
call_count(M::Wrapper) = call_count(base_method(M))

iter_count(M::OptBuffer) = -1
call_count(M::OptBuffer) = -1

"""
    convstat(M::Wrapper)

For a converged state, return the statistics in the form of `NamedTuple`
  `(converged = true/false, argument, gradient, iterations, calls)`.
  Negative values of `calls` or `iterations` mean that the number has not been tracked.
"""
function convstat(M::Wrapper)
    converged = conv_success(M)
    argument = argumentvec(M)
    gradient = gradientvec(M)
    iterations = iter_count(M)
    calls = call_count(M)
    @logmsg OptLogLevel """

    ==FINAL STATISTICS==
    # converged: $(converged)
    # final gradient: $(gradient)
    # final argument: $(argument)
    # previous argument: $(step_origin(M))
    # final func value: $(fnval(M))
    # previous func value: $(fnval_origin(M))
    # number of iterations: $(iterations)
    # number of function calls: $(calls)
    """
    return (;
        converged,
        argument,
        gradient,
        iterations,
        calls,
    )
end

"""
    BasicConvergenceStats

Wrapper type to provide basic stop conditions: magnitude of gradient is less than the
    specified value, objective function call count exceeds threshold, iteration count
    exceeds threshold.
"""
mutable struct BasicConvergenceStats{T<:AbstractOptBuffer,F} <: Wrapper
    descent::T
    convcond::F
    converged::Bool
    call_limit::Int
    call_count::Int
    iter_limit::Int
    iter_count::Int

    function BasicConvergenceStats(
        M::T;
        convcond::F,
        call_limit=typemax(Int),
        iter_limit=typemax(Int),
    ) where {T<:AbstractOptBuffer, F<:Base.Callable}
        return new{T, F}(M, convcond, false, call_limit, 0, iter_limit, 0)
    end
end

base_method(M::BasicConvergenceStats) = M.descent

function init!(fdf, M::BasicConvergenceStats, args...; kw...)
    M.call_count = 0
    M.iter_count = 0
    M.converged = false
    init!(fdf, M.descent, args...; kw...)
    return
end

function reset!(M::BasicConvergenceStats, args...; kw...)
    M.call_count = 0
    M.iter_count = 0
    M.converged = false
    reset!(M.descent, args...; kw...)
    return
end

function callfn!(fdf::F, M::BasicConvergenceStats, x, α, d) where {F}
    fg = callfn!(fdf, M.descent, x, α, d)
    M.call_count += 1
    return fg
end

function step!(fdf::F, M::BasicConvergenceStats; kw...) where {F}
    s = step!(fdf, M.descent; kw...)
    M.iter_count += 1
    return s
end

function stopcond(M::BasicConvergenceStats)
    base = base_method(M)

    x, xpre = argumentvec(base), step_origin(base)
    y, ypre = fnval(base), fnval_origin(base)
    g = gradientvec(base)
    M.converged = M.convcond(x, xpre, y, ypre, g)

    if M.converged
        return true
    elseif M.call_count >= M.call_limit
        return true
    elseif M.iter_count >= M.iter_limit
        return true
    else
        return stopcond(base)
    end
end

function conv_success(M::BasicConvergenceStats)
    base = base_method(M)
    if M.converged
        return true
    else
        return conv_success(base)
    end
end

@inline iter_count(M::BasicConvergenceStats) = M.iter_count
@inline call_count(M::BasicConvergenceStats) = M.call_count

"""
    ConstrainStepSize

Wrapper type to limit step sizes attempted in optimization,
given a function `(origin, direction) -> max_step`.
"""
struct ConstrainStepSize{F, T<:AbstractOptBuffer} <: Wrapper
    descent::T
    constraint::F

    function ConstrainStepSize(constr::F, optbuf::T) where {F,T<:AbstractOptBuffer}
        return new{F,T}(optbuf, constr)
    end
end

ConstrainStepSize(M::AbstractOptBuffer) = ConstrainStepSize(infstep, M)

base_method(M::ConstrainStepSize) = M.descent

function init!(fdf, M::ConstrainStepSize, args...; kw...)
    init!(fdf, M.descent, args...; constrain_step=M.constraint, kw...)
    return
end

step!(optfn!, M::ConstrainStepSize) = step!(optfn!, M.descent; constrain_step=M.constraint)

struct OptFunc{F<:Base.Callable,M<:AbstractOptBuffer}<:Function
    fdf::F
    buffer::M
end

function (optfn::OptFunc)(x, α, d)
    y, g = callfn!(optfn.fdf, optfn.buffer, x, α, d)
    @logmsg OptLogLevel "$(join(x .+ α .* d, ' ')) $y $(join(g, ' '))"
    y, g
end
