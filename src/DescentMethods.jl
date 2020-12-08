module DescentMethods

using LinearAlgebra

include("abstract_types.jl")
include("wrappers.jl")
include("linesearch.jl")
include("grad_descent.jl")
include("rate_descent.jl")
include("optimize.jl")
include("conjgrad.jl")

end # module
