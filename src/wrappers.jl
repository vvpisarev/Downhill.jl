export CountCalls, CountIters

function base_method(::DescentMethod) end

argumentvec(M::Wrapper) = argumentvec(base_method(M))
gradientvec(M::Wrapper) = gradientvec(base_method(M))
step_origin(M::Wrapper) = step_origin(base_method(M))

__step_init!(M::Wrapper, optfn!) = __step_init!(base_method(M), optfn!)
__descent_dir!(M::Wrapper) = __descent_dir!(base_method(M))
__compute_step!(M::Wrapper, args...; kw...) = __compute_step!(base_method(M), args...; kw...)

callfn!(M::Wrapper, fdf, x, α, d) = callfn!(base_method(M), fdf, x, α, d)

function init!(M::Wrapper, args...; kw...)
    init!(base_method(M), args...; kw...)
    return
end

function reset!(M::Wrapper, args...; kw...)
    reset!(base_method(M), args...; kw...)
    return
end

stopcond(M::Wrapper) = stopcond(base_method(M))
@inline stopcond(M::CoreMethod) = false

conv_success(M::Wrapper) = conv_success(base_method(M))
@inline conv_success(M::CoreMethod) = false

iter_count(M::Wrapper) = iter_count(base_method(M))
call_count(M::Wrapper) = call_count(base_method(M))

iter_count(M::CoreMethod) = -1
call_count(M::CoreMethod) = -1

convstat(M::Wrapper) = (converged = conv_success(M),
                        argument = argumentvec(M),
                        iterations = iter_count(M),
                        calls = call_count(M)
                       )

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

function callfn!(M::LimitCalls, fdf, x, α, d)
    fg = callfn!(M.descent, fdf, x, α, d)
    M.call_count += 1
    return fg
end

stopcond(M::LimitCalls) = M.call_count < M.call_limit ? stopcond(M.descent) : true
call_count(M::LimitCalls) = M.call_count

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

function reset!(M::LimitIters, args...; iter_limit = M.iter_limit, kw...)
    M.iter_count = 0
    M.iter_limit = iter_limit
    reset!(M.descent, args...; kw...)
    return
end

callfn!(M::LimitIters, fdf, x, α, d) = callfn!(M.descent, fdf, x, α, d)

function __compute_step!(M::LimitIters, args...; kw...)
    s = __compute_step!(M.descent, args...; kw...)
    M.iter_count += 1
    return s
end

stopcond(M::LimitIters) = M.iter_count < M.iter_limit ? stopcond(M.descent) : true

@inline iter_count(M::LimitIters) = M.iter_count

struct LimitStepSize{T<:DescentMethod, F} <: Wrapper
    descent::T
    maxstep::F
end

LimitStepSize(M::DescentMethod) = LimitStepSize(M, (x, d)->convert(eltype(d), Inf))

base_method(M::LimitStepSize) = M.descent

function __compute_step!(M::LimitStepSize, optfn!, d, maxstep = convert(eltype(d), Inf))
    maxlstep = min(maxstep,
                   M.maxstep(step_origin(M.descent), d)
                  )
    __compute_step!(M.descent, optfn!, d, maxlstep)
end