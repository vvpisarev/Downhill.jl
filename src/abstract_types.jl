export AbstractOptBuffer, OptBuffer, Wrapper

abstract type AbstractOptBuffer end

abstract type OptBuffer <: AbstractOptBuffer end

abstract type Wrapper <: AbstractOptBuffer end

const OptLogLevel = LogLevel(-10)
const LSLogLevel = LogLevel(-20)

# Default getters of `OptBuffer` fields
@inline gradientvec(M::OptBuffer) = M.g
@inline argumentvec(M::OptBuffer) = M.x
@inline step_origin(M::OptBuffer) = M.xpre
@inline fnval(M::OptBuffer) = M.y
@inline fnval_origin(M::OptBuffer) = M.ypre
