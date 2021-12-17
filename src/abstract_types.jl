export AbstractOptBuffer, OptBuffer, Wrapper

abstract type AbstractOptBuffer end

abstract type OptBuffer <: AbstractOptBuffer end

abstract type Wrapper <: AbstractOptBuffer end

const OptLogLevel = LogLevel(-10)
const LSLogLevel = LogLevel(-20)
