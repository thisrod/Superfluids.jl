# Superfluids user manual

This is a library to solve the Gross-Pitaeveskii equation and the
Bogoliubov-de Gennes eigenproblem, and plot the resulting order
parameters.  It is currently most fully developed for the 2-dimensional
case, but the intention is to support 1 and 3 dimensions as well.

This library is not intended to implement any numerical methods.
Those should be in general packages such as `Optim`,
`DifferentialEquations` and `IterativeSolvers`.  The aim is to
configure those packages to best handle superfluid problems, and
let them at it.  Enhancements to the methods should be contributed
upstream.

Well, that's the ideal.  Currently, the julia system lacks some
parts we need, especially when it comes to discretisations and
transverse derivatives.  Those are currently part of this package,
but the intention is to get rid of them and use a general solution
as soon as an adequate one becomes available.

## Defaults

Almost every function in the Superfluids library must be passed a
`Superfluid`, or at least a `Discretisation`.  To avoid unnecessary
typing, these arguments are optional, and there is a mechanism
borrowed from `Plots.jl` to specify defaults.

```@docs
default
default!
```

Other arguments are in the same boat:

* `g_tol`  The residual to which order parameters are relaxed.

* `dt`  The time step for solving the GPE


## Wave functions

Everything is normalised with norm(q) = 1, no cleverness about discretisation.