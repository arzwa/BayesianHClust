# Arthur Zwaenepoel 2021
module BayesianHClust

using Distributions, Parameters, SpecialFunctions, StatsFuns, StatsBase
using Memoize, NewickTree, RecipesBase

export BHClust, bhclust, dirichlet_cat_logpdf, dirichlet_mult_logpdf, cuttree

"""
    BHClust

Bayesian hierarchical clustering model. 

- `α` is the concentration parameter for the Dirichlet process (or CRP) prior. 

- `logdensity` is a function taking a single argument `x` being the sufficient
   statistics for computing the posterior density (so this will typically be a
   conjugate prior - likelihood function, such as the Dirichlet-multinomial or
   Inverse-Wishart-Normal density) 
 
- `mergedata` is a function that combines the data from two nodes in the tree to
   form the data (typically sufficient statistics for `logdensity`) at an internal
   node.
"""
struct BHClust{T}
    α::T
    logdensity::Function   # conjugate prior-likelihood
    mergedata ::Function   # function to merge data from two nodes
end

"""
    ClusterNode

A node in a Bayesian hierarchical clustering tree. 
"""
struct ClusterNode{T,V}
    n::Int
    d::T  # log(d)
    ℓ::T  # marginal loglikelihood P(Dₖ|Tₖ)
    r::T  # posterior probability of the forming a single cluster
    x::V  # data (summary statistic)
end

Base.show(io::IO, n::ClusterNode) = 
    write(io, "ClusterNode(p=$(round(exp(n.r), digits=2)), ℓ=$(n.ℓ))")

# initial nodes
ClusterNode(d, ℓ, x) = ClusterNode(1, d, ℓ, 0., x)  

function bhclust(X, model::BHClust)
    tojoin, P, M = initialize(X, model)
    k = length(tojoin) + 1
    while length(tojoin) > 1
        i, j = Tuple(argmax(P))
        newnode = join!(tojoin, P, M, i, j, k)
        update_matrices!(k, P, M, tojoin, model)
        k += 1
    end
    return tojoin[k-1]
end

function initnodes(X::AbstractMatrix, model)
    @unpack α, logdensity = model
    return [ClusterNode(log(α), logdensity(X[i,:]), X[i,:]) for i=1:size(X,1)]
end

function initnodes(X::AbstractVector, model)
    @unpack α, logdensity = model
    return [ClusterNode(log(α), logdensity(X[i]), X[i]) for i=1:size(X,1)]
end

function initialize(X, model)
    nodes = initnodes(X, model)
    n = length(nodes)
    tojoin = Dict(i=>Node(i, nodes[i]) for i=1:n)
    nmatrix = similar(nodes, 2n - 1, 2n -1)
    pmatrix = fill(-Inf, 2n - 1, 2n - 1)
    Threads.@threads for i=1:n
        for j=1:i-1
            p, node = compute_merge(nodes[i], nodes[j], model)
            pmatrix[i, j] = p
            nmatrix[i, j] = node
        end
    end
    tojoin, pmatrix, nmatrix
end

function join!(tojoin, P, M, i, j, k)
    P[i,:] .= -Inf   # nodes i and j should no longer be considered
    P[:,i] .= -Inf
    P[j,:] .= -Inf
    P[:,j] .= -Inf
    newnode = Node(k, M[i,j])
    push!(newnode, tojoin[i])
    push!(newnode, tojoin[j])
    delete!(tojoin, i)
    delete!(tojoin, j)
    tojoin[k] = newnode
end

function update_matrices!(k, P, M, tojoin, model)
    nk = tojoin[k]
    idx = collect(eachindex(tojoin))
    Threads.@threads for j=1:length(idx)
        i = idx[j]
        i == k && continue
        p, node = compute_merge(tojoin[i].data, nk.data, model)
        P[i,k] = p
        M[i,k] = node
    end
end

function compute_merge(n1, n2, model)
    @unpack mergedata, logdensity, α = model
    # log(d) and log(π)
    nk = n1.n + n2.n
    gα = log(α) + loggamma(nk)
    dk = logaddexp(gα, n1.d + n2.d)
    πk = gα - dk
    # single cluster marginal
    x  = mergedata(n1.x, n2.x)
    p1 = πk + logdensity(x) 
    # two cluster marginal
    p2 = log1mexp(πk) + n1.ℓ + n2.ℓ
    # complete marginal
    ℓk = logaddexp(p1, p2) 
    # log posterior probability of merge
    rk = p1 - ℓk
    @assert rk <= 0. "$rk"  # it is a posterior probability
    return rk, ClusterNode(nk, dk, ℓk, rk, x)
end

function dirichlet_mult_logpdf(xs, α::Real) 
    n = sum(xs)
    K = length(xs) 
    Σ = K * α
    l = loggamma(n + 1) + loggamma(Σ) - loggamma(n + Σ) - K*loggamma(α)
    for x in xs
        l += loggamma(x + α) - loggamma(x + 1.)
    end
    return l
end

function dirichlet_cat_logpdf(xs, α::Real)
    n = sum(xs)
    K = length(xs) 
    Σ = K * α
    l = loggamma(Σ) - K*loggamma(α) - loggamma(n + Σ) 
    for x in xs
        l += loggamma(x + α)
    end
    return l
end

function dirichlet_cat_logpdf(xs, α::AbstractVector{T}) where T
    n = sum(xs)
    K = length(xs)
    Σ = sum(α)
    l = loggamma(Σ) - sum(loggamma.(α))
    Z = zero(T)
    for i in eachindex(xs)
        l += loggamma(xs[i] + α[i])
        Z += xs[i] + α[i]
    end
    return l - loggamma(Z)
end 


"""
    cuttree(node, thresh=0.5)

Cut the tree at nodes where the posterior probability for the merged hypothesis
is smaller then `thresh`.
"""
function cuttree(node, thresh=0.5)
    function walk(node)
        isleaf(node) && return []
        below = [walk(c) for c in children(node)]
        a = node.data.r < log(thresh)
        if a && isempty(below[1]) && isempty(below[2])
            return [node[1], node[2]]
        elseif isempty(below[1]) && isempty(below[2])
            return below[1]
        elseif isempty(below[1]) 
            return vcat(node[1], below[2])
        elseif isempty(below[2]) 
            return vcat(below[1], node[2])
        else
            return vcat(below...)
        end
    end
    cuts = walk(node)
    labels = clusterlabels(cuts)
    return cuts, labels
end

function clusterlabels(cuts)
    clusters = map(getleaves, cuts)
    sort!(clusters, by=length, rev=true)
    labels = zeros(Int, length(vcat(clusters...)))
    for (k, cluster) in enumerate(clusters)
        ids = id.(cluster)
        for i in ids
            labels[i] = k
        end
    end
    return labels
end

@recipe function f(tree::Node{I,<:ClusterNode}; thresh=0.5) where I 
    cuts, labels = cuttree(tree, thresh)    
    nmap = indexmap(id.(postwalk(tree))) 
    x, y = NewickTree.treepositions(tree, false)
    colors = zeros(Int, 1, size(x,2))
    layout --> (1,2)
    for (i,cluster) in enumerate(cuts)  
        for n in prewalk(cluster)           
            colors[1,nmap[id(n)]] = i       
            # these are the same as the cluster labels
            # (also by enumeration of cuts) 
        end
    end
    @series begin
        seriescolor --> colors
        linewidth --> 2
        legend --> false
        yticks --> false
        xticks --> false
        grid --> false
        subplot := 2
        framestyle --> :none
        x, y
    end
    @series begin
        subplot := 1
        legend --> false
        seriescolor --> :black
        framestyle --> :box
        linewidth --> 1
        xguide --> "\$r\$"
        yguide --> "# clusters"
        xs = 0.0:0.001:1.
        ys = map(xs) do thresh
            cuts, labels = cuttree(tree, thresh)
            length(cuts)
        end
        xs, ys
    end
end


end # module
