Implementation of the Bayesian Hierarchical clustering method of
[Heller & Ghahramani](https://dl.acm.org/doi/pdf/10.1145/1102351.1102389)
Currently with focus on Dirichlet-Categorical/Dirichlet-Multinomial
conjugate models, but the implementation should be general enough to
admit other models.

```
Heller, K. A., & Ghahramani, Z. (2005, August). 
Bayesian hierarchical clustering. 
In Proceedings of the 22nd international conference on Machine learning (pp. 297-304).
```

Note this is software written for my own research purposes, it is
**still experimental and needs more tests!**.

## Example

```julia
using BayesianHClust, Distributions, Plots
ps = rand(Dirichlet(3, 0.5), 3)
Xs = map(p->rand(Multinomial(50, Array(p)), 5), eachcol(ps))
X  = permutedims(hcat(Xs...))  
    
clustalg = BHClust(.1, x->dirichlet_cat_logpdf(x, 1.), (x,y)->x .+ y)
tree = bhclust(X, clustalg)
plot(tree)
```

