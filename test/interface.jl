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

    @testset "argumentvec" for descent in descent_methods
        @test size(argumentvec(descent)) == size(init_vec)
    end

    @testset "gradientvec" for descent in descent_methods
        @test size(gradientvec(descent)) == size(init_vec)
    end

    @testset "step_origin" for descent in descent_methods
        @test size(step_origin(descent)) == size(init_vec)
    end
end