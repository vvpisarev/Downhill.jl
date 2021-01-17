@testset "Constructors from Float vectors" begin
    @test SteepestDescent([1.0, 0]) isa CoreMethod
    @test HyperGradDescent([1.0, 0]) isa CoreMethod
    @test FixedRateDescent([1.0, 0]) isa CoreMethod
    @test CGDescent([1.0, 0]) isa CoreMethod
    @test BFGS([1.0, 0]) isa CoreMethod
end

@testset "Constructors from Int vectors" begin
    @test SteepestDescent([1, 0]) isa CoreMethod
    @test HyperGradDescent([1, 0]) isa CoreMethod
    @test FixedRateDescent([1, 0]) isa CoreMethod
    @test CGDescent([1, 0]) isa CoreMethod
    @test BFGS([1, 0]) isa CoreMethod
end

@testset "Constructors from Rational vectors" begin
    @test SteepestDescent([1//1, 0]) isa CoreMethod
    @test HyperGradDescent([1//1, 0]) isa CoreMethod
    @test FixedRateDescent([1//1, 0]) isa CoreMethod
    @test CGDescent([1//1, 0]) isa CoreMethod
    @test BFGS([1//1, 0]) isa CoreMethod
end