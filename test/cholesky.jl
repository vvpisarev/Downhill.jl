@testset "Modified Cholesky decomposition" begin
    @testset "SPD matrix" begin
        dim = 10
        @testset for ntest in 1:15
            a = rand(test_rng, Float64, (dim, dim))
            a = a' * a
            achol = mcholesky!(copy(a))
            # For SPD matrices, must be the same as the usual Cholesky decomposition
            @test isapprox(achol.U' * achol.U, a; rtol = 1e-4)
        end
    end

    @testset "Generic symmetric matrix" begin
        dim = 10
        @testset for ntest in 1:15
            a = rand(test_rng, Float64, (dim, dim))
            a .+= a'
            achol = mcholesky!(copy(a))
            modcholprod = achol.U' * achol.U
            # checking that the decomposition differs from the original matrix
            # only in diagonal elements
            modcholprod .-= Diagonal(modcholprod)
            a .-= Diagonal(a)
            @test isapprox(UpperTriangular(modcholprod), UpperTriangular(a); rtol = 1e-4)
        end
    end
end
