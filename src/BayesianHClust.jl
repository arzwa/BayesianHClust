module BayesianHClust

using Distributions, Parameters, SpecialFunctions, StatsFuns, StatsBase
using Memoize, NewickTree, RecipesBase

export BHClust, bhclust, dirichlet_cat_logpdf, cuttree

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

function initialize(X::AbstractMatrix, model)
    @unpack α, logdensity = model
    return [Node(i, ClusterNode(log(α), logdensity(X[i,:]), X[i,:])) for i=1:size(X,1)]
end

function initialize(X::AbstractVector, model)
    @unpack α, logdensity = model
    return [Node(i, ClusterNode(log(α), logdensity(X[i]), X[i])) for i=1:size(X,1)]
end

"""
    bhclust(X, model::BHClust)

Grow the Bayesian hierarchical clustering tree for data set X.
"""
function bhclust(X, model)
    nodes = initialize(X, model)
    k = length(nodes) + 1
    while length(nodes) > 1
        nodes = merge(nodes, model, k)
        k += 1
    end
    return nodes[1]
end

function merge(nodes, model, k) 
    N = length(nodes)
    best = (-Inf, nothing, 0, 0)
    for i=2:N, j=1:i-1
        p, n = compute_merge(nodes[i].data, nodes[j].data, model)  
        # returns a clusternode, memoized
        best = p > best[1] ? (p, n, i, j) : best
    end
    # join best
    newnode = Node(k, best[2])
    push!(newnode, nodes[best[3]])
    push!(newnode, nodes[best[4]])
    # update nodes to join -- this is ugly
    newnodes = similar(nodes, N-1)
    newnodes[1] = newnode
    z = 2
    for (i,n) in enumerate(nodes)
        (i == best[3] || i == best[4]) && continue
        newnodes[z] = n
        z += 1
    end
    return newnodes
end

# this is for the Dirichlet-multinomial model
# I'm a bit lazy and use Memoization instead of more carefully storing
# computations that have already been done...
@memoize function compute_merge(n1, n2, model)
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

#function dirichlet_mult_logpdf(xs, α::Real) 
#    n = sum(xs)
#    K = length(xs) 
#    Σ = K * α
#    l = loggamma(n + 1) + loggamma(Σ) - loggamma(n + Σ) - K*loggamma(α)
#    for x in xs
#        l += loggamma(x + α) - loggamma(x + 1.)
#    end
#    return l
#end

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
    for (i,cluster) in enumerate(cuts)  
        for n in prewalk(cluster)           
            colors[1,nmap[id(n)]] = i       
            # these are the same as the cluster labels
            # (also by enumeration of cuts) 
        end
    end
    seriescolor --> colors
    linewidth --> 1
    legend --> false
    yticks --> false
    xticks --> false
    grid --> false
    framestyle --> :none
    x, y
end


end # module
