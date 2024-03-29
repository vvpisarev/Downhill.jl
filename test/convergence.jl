@testset "Convergence on the Rosenbrock function" begin
    dim = 2
    x0 = fill(-1, dim)
    @testset "By gradient norm (default)" begin
        for method in OPT_TYPES
            opt = method(x0)
            optresult = optimize!(
                rosenbrock!, opt, x0;
                maxiter=1000,
                constrain_step=(x0,d)->Inf,
            )
            @testset "$method" begin
                if opt isa FixedRateDescent
                    @test_broken optresult.converged
                    @test optresult.argument ≈ [1, 1] rtol=0.05
                else
                    @test optresult.converged
                    @test optresult.argument ≈ [1, 1] rtol=0.05
                end
            end
        end
    end
    @testset "By x - xpre" begin
        for method in OPT_TYPES
            opt = method(x0)
            optresult = optimize!(
                rosenbrock!, opt, x0;
                maxiter=1000,
                constrain_step=(x0,d)->Inf,
                convcond=(x, xpre, y, ypre, g) -> norm(x - xpre, 2) ≤ 1e-6,
            )
            @testset "$method" begin
                if any(T -> opt isa T, (HyperGradDescent, FixedRateDescent, MomentumDescent, NesterovMomentum))
                    @test_broken optresult.converged
                    @test optresult.argument ≈ [1, 1] rtol=0.05
                else
                    @test optresult.converged
                    @test optresult.argument ≈ [1, 1] rtol=0.05
                end
            end
        end
    end
    @testset "By y - ypre" begin
        for method in OPT_TYPES
            opt = method(x0)
            optresult = optimize!(
                rosenbrock!, opt, x0;
                maxiter=1000,
                constrain_step=(x0,d)->Inf,
                convcond=(x, xpre, y, ypre, g) -> isapprox(y, ypre; atol=1e-6),
            )
            @testset "$method" begin
                if any(T -> opt isa T, (HyperGradDescent, FixedRateDescent, MomentumDescent, NesterovMomentum))
                    @test_broken optresult.converged
                    @test optresult.argument ≈ [1, 1] rtol=0.05
                else
                    @test optresult.converged
                    @test optresult.argument ≈ [1, 1] rtol=0.05
                end
            end
        end
    end
end
