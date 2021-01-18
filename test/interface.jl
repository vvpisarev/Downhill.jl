@testset "Common interface functions"
    init_vec = [1.0, 0]
    steepest = SteepestDescent(init_vec)
    hypergrad = HyperGradDescent(init_vec)
    fixedrate = FixedRateDescent(init_vec)
    conjgrad = CGDescent(init_vec)
    bfgs = BFGS(init_vec)
    @testset "argumentvec" for descent in (steepest,
                                           hypergrad,
                                           fixedrate,
                                           conjgrad,
                                           bfgs)
        @test size(argumentvec(descent)) == size(init_vec)
    end

    @testset "gradientvec" for descent in (steepest,
                                           hypergrad,
                                           fixedrate,
                                           conjgrad,
                                           bfgs)
        @test size(gradientvec(descent)) == size(init_vec)
    end

    @testset "step_origin" for descent in (steepest,
                                           hypergrad,
                                           fixedrate,
                                           conjgrad,
                                           bfgs)
        @test size(step_origin(descent)) == size(init_vec)
    end
end