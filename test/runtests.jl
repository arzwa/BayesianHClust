using Test, Serialization, Distributions, NewickTree, Plots
using BayesianHClust, Random

@testset "Simple Dirichlet-Multinomial problem" begin
    Random.seed!(321)
    p1 = [0.1, 0.1, 0.8]
    p2 = [0.1, 0.8, 0.1]
    p3 = [0.8, 0.1, 0.1]
    ps = repeat([p1, p2, p3], 5)
    X = permutedims(mapreduce(p->rand(Multinomial(50, p)), hcat, ps))
    α = .1
    β = 10.
    m = BHClust(α, x->dirichlet_cat_logpdf(x, β), (x,y)->x .+ y)
    t = bhclust(X, m)     
    clusters, labels = cuttree(t)
    @test length(clusters) == 3
    for n in clusters
        leaves = sort(id.(getleaves(n)))
        expected = collect(leaves[1]:3:length(ps))
        @test all(leaves .== expected)
    end
end
