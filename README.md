# Superfluids.jl

A wrapper for Julia's numerical libraries, to compute the eigenstates and dynamics of the Gross-Pitaevskii and the Bogoliubov-de Gennes equations.

Remember: just encapsulate what you have, don't immediately add intervals or anything fancy.

A `Superfluid` stores `V`, as a function of coordinates, `Nc`, `C`, and `Ω`.  The `Nc` can be `missing` in case of a uniform fluid or a pure order parameter problem.

A `Discretisation` includes a domain.  It has a `laplacian` method that returns the laplacian as a Tensar, `angular` that gives the angular momentum operator, and `position` that evaluates a function of coordinates and returns a tensar diagonal in the position basis.

## Defaults

`default` and `default!`

Clash with plots

`Superfluid` and `Discretisation` special cases