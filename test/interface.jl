@testset "Interfaces" begin
    @test SteepestDescent([1, 0]) isa CoreMethod
    @test HyperGradDescent([1, 0]) isa CoreMethod
    @test FixedRateDescent([1, 0]) isa CoreMethod
end