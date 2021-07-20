using Test, Serialization, Distributions, NewickTree, Plots
using BayesianHClust, Random

@testset "Simple Dirichlet-Multinomial problem" begin
    Random.seed!(321)
    p1 = [0.1, 0.1, 0.8]
    p2 = [0.1, 0.8, 0.1]
    p3 = [0.8, 0.1, 0.1]
    ps = repeat([p1, p2, p3], 5)
    X = permutedims(mapreduce(p->rand(Multinomial(50, p)), hcat, ps))
    α = 1.
    β = 1.
    m = BHClust(α, x->dirichlet_mult_logpdf(x, β), (x,y)->x .+ y)
    t = bhclust(X, m)     
    clusters = cuttree(t)
    @test length(clusters) == 3
    for n in clusters
        leaves = sort(id.(getleaves(n)))
        expected = collect(leaves[1]:3:length(ps))
        @test all(leaves .== expected)
    end
end

x, y = NewickTree.treepositions(t, true, false)
z = reshape([n.data.r > log(0.5) ? :black : :red for n in postwalk(t)], 1, size(x,2))
plot(x, y, color=z, legend=false, grid=false)

s = "/home/arzwa/research/drosera/revision-bis/17.segmentation/dca-dre.map-og.jls"
X = deserialize(s)
idx = filter(i->sum(X[:,i]) > 1, 1:size(X,2))
X = X[:,idx]

ℓfun = x->dirichlet_mult_logpdf(x, .1)
dfun = (x,y)->x .+ y 
t = bhclust(X, BHClust(1., ℓfun, dfun))

x, y = NewickTree.treepositions(t, true, false)

using ColorSchemes

c = ColorSchemes.viridis
z = reshape([c[exp(n.data.r)] for n in postwalk(t)], 1, size(x,2))
o = id.(filter(isleaf, postwalk(t)))
plot(plot(x, y, color=z, legend=false, grid=false, 
          xlim=(0,length(o)), xticks=false, 
          xshowaxis=false, ytick=false, yshowaxis=false), 
     heatmap(permutedims(log.(X[o,:])), 
             yshowaxis=false, yticks=false,
             color=:binary, framestyle=:box, 
             colorbar=false, xlim=(0,length(o))),
     layout=grid(2,1,heights=[0.3, 0.7]))
