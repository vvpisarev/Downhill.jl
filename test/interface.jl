using DescentMethods: argumentvec, gradientvec, step_origin

@testset "Common interface functions" begin
    init_vec = [1.0, 0]
    steepest = SteepestDescent(init_vec)
    hypergrad = HyperGradDescent(init_vec)
    fixedrate = FixedRateDescent(init_vec)
    conjgrad = CGDescent(init_vec)
    bfgs = BFGS(init_vec)

    descent_methods = (
        steepest,
        hypergrad,
        fixedrate,
        conjgrad,
        bfgs,
    )

    interface = (
        argumentvec,
        gradientvec,
        step_origin
    )
    @testset "$(string(interface_func))" for interface_func in interface
        @testset "$(typeof(descent).name)" for descent in descent_methods
            @test size(interface_func(descent)) == size(init_vec)
        end
    end
end