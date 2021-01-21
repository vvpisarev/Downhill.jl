@testset "Constructors for core methods" begin
    types = (
        SteepestDescent,
        HyperGradDescent,
        FixedRateDescent,
        CGDescent,
        BFGS
    )

    @testset "From Float vector" begin
        v = [1.0, 0]
        @testset "$T" for T in types
            @test T(v) isa CoreMethod
        end
    end

    @testset "From Int vector" begin
        v = [1, 0]
        @testset "$T" for T in types
            @test T(v) isa CoreMethod
        end
    end

    @testset "From Rational vector" begin
        v = [1//1, 0]
        @testset "$T" for T in types
            @test T(v) isa CoreMethod
        end
    end
end