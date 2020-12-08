export CountCalls, CountIters

mutable struct CountCalls{T<:DescentMethod}<:DescentMethod
    descent::T
    call_count::Int
end

CountCalls(M::DescentMethod) = CountCalls(M, 0)

function init!(M::CountCalls, args...; kw...)
    M.call_count = 0
    init!(M.descent, args...; kw...)
    return
end

function callfn!(M::CountCalls, fdf, x, α, d)
    fg = callfn!(M.descent, fdf, x, α, d)
    M.call_count += 1
    return fg
end

step!(M::CountCalls, args...; kw...) = step!(M.descent, args...; kw...)

isconverged(M::CountCalls, args...; kw...) = isconverged(M.descent, args...; kw...)

argumentvec(M::CountCalls) = argumentvec(M.descent)
gradientvec(M::CountCalls) = gradientvec(M.descent)

mutable struct CountIters{T<:DescentMethod}<:DescentMethod
    descent::T
    iter_count::Int
end

CountIters(M::DescentMethod) = CountIters(M, 0)

function init!(M::CountIters, args...; kw...)
    M.iter_count = 0
    init!(M.descent, args...; kw...)
    return
end

callfn!(M::CountIters, fdf, x, α, d) = callfn!(M.descent, fdf, x, α, d)

function step!(M::CountIters, args...; kw...)
    step!(M.descent, args...; kw...)
    M.iter_count += 1
end

isconverged(M::CountIters, args...; kw...) = isconverged(M.descent, args...; kw...)

argumentvec(M::CountIters) = argumentvec(M.descent)
gradientvec(M::CountIters) = gradientvec(M.descent)