export DescentMethod, CoreMethod, Wrapper

abstract type DescentMethod end

function step!(M::DescentMethod, optfn!, maxstep = Inf)
    __step_init!(M, optfn!)
    d = __descent_dir!(M)
    __compute_step!(M, optfn!, d, maxstep)
end

abstract type CoreMethod <: DescentMethod end

abstract type Wrapper <: DescentMethod end