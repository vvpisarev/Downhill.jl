module DescentMethods

using LinearAlgebra

include("utils.jl")
include("abstract_types.jl")
include("wrappers.jl")
include("linesearch.jl")
include("grad_descent.jl")
include("rate_descent.jl")
include("hypergradient.jl")
include("quasinewton.jl")
include("chol_bfgs.jl")
include("optimize.jl")
include("conjgrad.jl")

end # module
