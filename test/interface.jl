@testset "Common interface functions" begin
    init_vec = [1.0, 0.0]

    descent_methods = map(T -> T(init_vec), OPT_TYPES)

    interface_vec = (
        argumentvec,
        gradientvec,
        step_origin,
    )
    @testset "$(string(interface_func))" for interface_func in interface_vec
        @testset "$(typeof(descent).name)" for descent in descent_methods
            @test size(interface_func(descent)) == size(init_vec)
        end
    end
    interface_scalar = (
        fnval,
        fnval_origin,
    )
    @testset "$(string(interface_func))" for interface_func in interface_scalar
        @testset "$(typeof(descent).name)" for descent in descent_methods
            @test interface_func(descent) isa Number
        end
    end
end

@testset "Solver objects" begin
    init_vec = [1.0, 0.0]

    descent_methods = map(T -> T(init_vec), OPT_TYPES)

    @testset "Solvers" begin
        @testset "$(typeof(descent).name)" for descent in descent_methods
            opt = Downhill.solver(
                descent;
                gtol=1e-3,
                maxiter=100,
                maxcalls=1000,
                constrain_step = (x, d) -> Inf,
            )
            @test optimize!(rosenbrock!, opt, init_vec) isa NamedTuple
        end
    end
end
