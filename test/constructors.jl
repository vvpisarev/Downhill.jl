@testset "Constructors for core methods" begin
    @testset "From Float vector" begin
        v = [1.0, 0]
        @testset "$T" for T in OPT_TYPES
            @test T(v) isa T
        end
    end

    @testset "From Int vector" begin
        v = [1, 0]
        @testset "$T" for T in OPT_TYPES
            @test T(v) isa T
        end
    end

    @testset "From Rational vector" begin
        v = [1//1, 0]
        @testset "$T" for T in OPT_TYPES
            @test T(v) isa T
        end
    end
end
