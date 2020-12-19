export DescentMethod, CoreMethod, Wrapper

abstract type DescentMethod end

abstract type CoreMethod <: DescentMethod end

abstract type Wrapper <: DescentMethod end