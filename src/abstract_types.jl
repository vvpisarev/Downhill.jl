export AbstractOptBuffer, OptBuffer, Wrapper

"""
    AbstractOptBuffer

Abstract type for buffers used to implement optimizations routines (including stopping
    criteria etc.)
"""
abstract type AbstractOptBuffer end

"""
    OptBuffer

Abstract type for structs designed to store the data required for core optimization logic.
"""
abstract type OptBuffer <: AbstractOptBuffer end

"""
    Wrapper

Abstract type for wrappers around core buffers meant for auxiliary purposes (stopping
    conditions, logging etc.)
"""
abstract type Wrapper <: AbstractOptBuffer end

const OptLogLevel = LogLevel(-10)
const LSLogLevel = LogLevel(-20)

# Default getters of `OptBuffer` fields

"""
    gradientvec(M::AbstractOptBuffer)

Return the gradient vector at the end of the optimization step.
"""
@inline gradientvec(M::OptBuffer) = M.g

"""
    argumentvec(M::AbstractOptBuffer)

Return the argument vector at the end of the optimization step.
"""
@inline argumentvec(M::OptBuffer) = M.x

"""
    step_origin(M::AbstractOptBuffer)

Return the argument vector at the start of the optimization step.
"""
@inline step_origin(M::OptBuffer) = M.xpre

"""
    fnval(M::AbstractOptBuffer)

Return the objective function value at the end of the optimization step.
"""
@inline fnval(M::OptBuffer) = M.y

"""
    fnval_origin(M::AbstractOptBuffer)

Return the objective function value at the start of the optimization step.
"""
@inline fnval_origin(M::OptBuffer) = M.ypre
