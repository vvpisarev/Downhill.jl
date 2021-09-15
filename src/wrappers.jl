function base_method(::DescentMethod) end

argumentvec(M::Wrapper) = argumentvec(base_method(M))
gradientvec(M::Wrapper) = gradientvec(base_method(M))
step_origin(M::Wrapper) = step_origin(base_method(M))

__descent_dir!(M::Wrapper) = __descent_dir!(base_method(M))

@inline callfn!(M::Wrapper, fdf, x, α, d) = callfn!(base_method(M), fdf, x, α, d)


function init!(M::Wrapper, args...; kw...)
    init!(base_method(M), args...; kw...)
    return
end

"""
    reset!(M::DescentMethod, args...; kwargs...)

Reset the solver parameters to the default (or to specific value - see the documentation for
    the specific types).

Each method has to implement a parameter-free `reset!(M)` method.
"""
function reset!(M::Wrapper, args...; kw...)
    reset!(base_method(M), args...; kw...)
    return
end

step!(M::Wrapper, fn::F; kw...) where {F} = step!(base_method(M), fn; kw...)

stopcond(M::Wrapper) = stopcond(base_method(M))
@inline stopcond(M::CoreMethod) = false

conv_success(M::Wrapper) = conv_success(base_method(M))
@inline conv_success(M::CoreMethod) = false

iter_count(M::Wrapper) = iter_count(base_method(M))
call_count(M::Wrapper) = call_count(base_method(M))

iter_count(M::CoreMethod) = -1
call_count(M::CoreMethod) = -1

"""
    convstat(M::Wrapper)

For a converged state, return the statistics in the form of `NamedTuple`
  `(converged = true/false, argument, iterations, calls)`. Negative values of `calls` or
  `iterations` mean that the number has not been tracked.
"""
function convstat(M::Wrapper)
    return (
        converged = conv_success(M),
        argument = argumentvec(M),
        iterations = iter_count(M),
        calls = call_count(M)
    )
end

"""
    StopByGradient

Wrapper type to stop optimization once the magnitude of gradient
is less than the specified value.
"""
struct StopByGradient{T<:DescentMethod, F} <: Wrapper
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
mutable struct LimitCalls{T<:DescentMethod}<:Wrapper
    descent::T
    call_limit::Int
    call_count::Int
end

LimitCalls(M::DescentMethod) = LimitCalls(M, typemax(Int), 0)
LimitCalls(M::DescentMethod, maxcalls::Integer) = LimitCalls(M, maxcalls, 0)

base_method(M::LimitCalls) = M.descent

function init!(M::LimitCalls, args...; kw...)
    M.call_count = 0
    init!(M.descent, args...; kw...)
    return
end

function reset!(M::LimitCalls, args...; call_limit, kw...)
    M.call_count = 0
    M.call_limit = call_limit
    reset!(M.descent, args...; kw...)
    return
end

function callfn!(M::LimitCalls, fdf::F, x, α, d) where {F}
    fg = callfn!(M.descent, fdf, x, α, d)
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
mutable struct LimitIters{T<:DescentMethod}<:Wrapper
    descent::T
    iter_limit::Int
    iter_count::Int
end

LimitIters(M::DescentMethod) = LimitIters(M, typemax(Int), 0)
LimitIters(M::DescentMethod, maxiters::Integer) = LimitIters(M, maxiters, 0)

base_method(M::LimitIters) = M.descent

function init!(M::LimitIters, args...; kw...)
    M.iter_count = 0
    init!(M.descent, args...; kw...)
    return
end

function reset!(M::LimitIters, args...; iter_limit=M.iter_limit, kw...)
    M.iter_count = 0
    M.iter_limit = iter_limit
    reset!(M.descent, args...; kw...)
    return
end


function step!(M::LimitIters, fdf::F; kw...) where {F}
    s = step!(M.descent, fdf; kw...)
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
struct ConstrainStepSize{T<:DescentMethod, F} <: Wrapper
    descent::T
    constraint::F
end

ConstrainStepSize(M::DescentMethod) = ConstrainStepSize(M, infstep)

base_method(M::ConstrainStepSize) = M.descent

function init!(M::ConstrainStepSize, args...; kw...)
    init!(M.descent, args...; constrain_step=M.constraint, kw...)
    return
end

step!(M::ConstrainStepSize, optfn!) = step!(M.descent, optfn!, constrain_step=M.constraint)

"""
    TrackPath

Wrapper type to dump the steps during the optimization.
"""
struct TrackPath{T<:DescentMethod,F<:IO} <: Wrapper
    descent::T
    file::F
end

base_method(M::TrackPath) = M.descent

function callfn!(M::TrackPath, fdf, x, α, d)
    print(M.file, join(x .+ α .* d, ' '))
    fg = callfn!(M.descent, fdf, x, α, d)
    y, g = fg
    println(M.file, ' ', y, ' ', join(g, ' '))
    return fg
end

function step!(M::TrackPath, args...; kw...)
    println(M.file)
    step!(M.descent, args...; kw...)
end
