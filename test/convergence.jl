@testset "Convergence on the Rosenbrock function" begin
    dim = 2
    x0 = fill(-1, dim)
    @testset for method in OPT_TYPES
        opt = method(x0)
        optresult = optimize!(
            rosenbrock!, opt, x0;
            maxiter=1000,
            constrain_step=(x0,d)->Inf,
        )
        @test isapprox(optresult.argument, [1, 1], rtol=0.05)
    end
end
