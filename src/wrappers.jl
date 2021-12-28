function base_method(::AbstractOptBuffer) end

argumentvec(M::Wrapper) = argumentvec(base_method(M))
gradientvec(M::Wrapper) = gradientvec(base_method(M))
step_origin(M::Wrapper) = step_origin(base_method(M))

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
    StopByGradient

Wrapper type to stop optimization once the magnitude of gradient
is less than the specified value.
"""
struct StopByGradient{T<:AbstractOptBuffer, F} <: Wrapper
    descent::T
    gtol::F
end

base_method(M::StopByGradient) = M.descent

function stopcond(M::StopByGradient)
    base = M.descent
    norm(gradientvec(base)) <= M.gtol ? true : stopcond(base)
end

function conv_success(M::StopByGradient)
    base = M.descent
    norm(gradientvec(base), Inf) <= M.gtol ? true : conv_success(base)
end

"""
    LimitCalls

Wrapper type to stop optimization once the number of
the objective function calls exceeds the specified value.
"""
mutable struct LimitCalls{T<:AbstractOptBuffer}<:Wrapper
    descent::T
    call_limit::Int
    call_count::Int
end

LimitCalls(M::AbstractOptBuffer) = LimitCalls(M, typemax(Int), 0)
LimitCalls(M::AbstractOptBuffer, maxcalls::Integer) = LimitCalls(M, maxcalls, 0)

base_method(M::LimitCalls) = M.descent

function init!(fdf, M::LimitCalls, args...; kw...)
    M.call_count = 0
    init!(fdf, M.descent, args...; kw...)
    return
end

function reset!(M::LimitCalls, args...; call_limit, kw...)
    M.call_count = 0
    M.call_limit = call_limit
    reset!(M.descent, args...; kw...)
    return
end

function callfn!(fdf::F, M::LimitCalls, x, α, d) where {F}
    fg = callfn!(fdf, M.descent, x, α, d)
    M.call_count += 1
    return fg
end

stopcond(M::LimitCalls) = M.call_count < M.call_limit ? stopcond(M.descent) : true
call_count(M::LimitCalls) = M.call_count

"""
    LimitIters

Wrapper type to stop optimization once the number of
the optimization iterations exceeds the specified value.
"""
mutable struct LimitIters{T<:AbstractOptBuffer}<:Wrapper
    descent::T
    iter_limit::Int
    iter_count::Int
end

LimitIters(M::AbstractOptBuffer) = LimitIters(M, typemax(Int), 0)
LimitIters(M::AbstractOptBuffer, maxiters::Integer) = LimitIters(M, maxiters, 0)

base_method(M::LimitIters) = M.descent

function init!(fdf, M::LimitIters, args...; kw...)
    M.iter_count = 0
    init!(fdf, M.descent, args...; kw...)
    return
end

function reset!(M::LimitIters, args...; iter_limit=M.iter_limit, kw...)
    M.iter_count = 0
    M.iter_limit = iter_limit
    reset!(M.descent, args...; kw...)
    return
end


function step!(fdf::F, M::LimitIters; kw...) where {F}
    s = step!(fdf, M.descent; kw...)
    M.iter_count += 1
    return s
end

stopcond(M::LimitIters) = M.iter_count < M.iter_limit ? stopcond(M.descent) : true

@inline iter_count(M::LimitIters) = M.iter_count

"""
    ConstrainStepSize

Wrapper type to limit step sizes attempted in optimization,
given a function `(origin, direction) -> max_step`.
"""
struct ConstrainStepSize{F, T<:AbstractOptBuffer} <: Wrapper
    descent::T
    constraint::F
end

ConstrainStepSize(M::AbstractOptBuffer) = ConstrainStepSize(M, infstep)

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
