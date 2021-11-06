# # Superfluids user manual
# 
# This is a library to solve the Gross-Pitaeveskii equation and the
# Bogoliubov-de Gennes eigenproblem, and plot the resulting order
# parameters.  It is currently most fully developed for the 2-dimensional
# case, but the intention is to support 1 and 3 dimensions as well.
# 
# This library is not intended to implement any numerical methods.
# Those should be in general packages such as `Optim`,
# `DifferentialEquations` and `IterativeSolvers`.  The aim is to
# configure those packages to best handle superfluid problems, and
# let them at it.  Enhancements to the methods should be contributed
# upstream.
# 
# Well, that's the ideal.  Currently, the julia system lacks some
# parts we need, especially when it comes to discretisations and
# transverse derivatives.  Those are currently part of this package,
# but the intention is to get rid of them and use a general solution
# as soon as an adequate one becomes available.
# 
# ## Overview
# 
# To start, define a 2-dimensional harmonic trap with atomic repulsion

using Superfluids, Plots
s = Superfluid{2}(500, (x, y) -> x^2 + y^2)

# and discretise it by a high order finite-difference formula on a
# moderate sized 66×66 grid

d = FDDiscretisation{2}(66, 0.3)

# Crop superfluid plots to the interesting part of the cloud, but leave other plots as is

Superfluids.default!(:xlims, (-5, 5))
Superfluids.default!(:ylims, (-5, 5))

# The ground state is the expected Thomas-Fermi cloud

ψ₀ = steady_state(s, d)
plot(d, ψ₀)
#hide savefig("sf001.png")

# ![plot of harmonic trap ground state cloud](sf001.png)

# ### GPE dynamics
# 
# The state `ψ₀` is simply an `Array{2}`, which can be given a momentum kick as follows

q = ψ₀ .* Superfluids.sample((x, y) -> exp(1im * (x - y / 2)), d)
plot(d, q)

# The dynamics can be solved as follows.  The range is the times
# to return the order paramter.

qs = Superfluids.integrate(s, d, q, 0:0.1:2π)
@animate for q in qs
    plot(d, q)
end
#hide gif(ans, "sf002.gif"; fps=3)

# ![animation of harmonic trap Kohn mode](sf002.gif)

# ### Bogoliubov de-Gennes modes
# 
# Find the Kohn mode statically

ωs, us, vs = bdg_modes(s, d, ψ₀, 0.0, 15, nev = 50)
Superfluids.bdgspectrum(s, d, ωs, us, vs, leg = :none)

# 
# Animate and compare to GPE solution
# 
# ### Vortices
# 
# A novel feature of Superfluids is the ability to relax the order parameter, while constraining the positions of the vortices.  For example, we can plot the energy landscape of a vortex lattice, as a function of the lattice spacing.

Ω = 0.4
uu = @. exp(2π*1im*(0:5)/6)
uu = [0, uu...]
rr = 0.5:0.5:4
qs = [steady_state(s, d; Ω, rvs=r*uu) for r in rr]
H = Superfluids.operators(s, d, :H) |> only
Es = [dot(q, H(q;Ω)) |> real for q in qs]
scatter(rr, Es)

# The function `steady_lattice` relaxes the vortices to a steady lattice.

... = steady_lattice(r -> r*uu, s, d, 2.0)

# Add a KT mode
# 
# Detect the vortex positions and plot their paths over time
# 
# ## Scope
# 
# Superfluids are a class of physical systems, including Bose-Einstein
# condensates, that have global coherence described by an order parameter ``\psi``.
# This package computes order parameters that satisfy the Gross-Pitaevskii
# equation, and related quantities such as Bogoliubov sound wave
# modes.  This section will outline the theory, to define the
# conventions used in the package.  For a full treatment, refer to
# Fetter & Waleka or Pethick & Smith.
# 
# The order parameter theory can be extended to a Monte-Carlo phase
# space treatment, that includes more quantum mechanical effects.
# The package does not currently provide explicit support for this,
# but it would be an obvious extension for the future.
# 
# # The GPE, divided by ``\hbar`` to make every term a rate of change, is
# ```math
# i{\partial\psi\over\partial t} = \left(-{1\over2}{\hbar\over m}\nabla^2 + {V\over\hbar} +
#     {C\over\hbar}|\psi|^2\right)\psi = {\cal L}\psi.
# ```
# Where ``V`` is an external potential, typically from a trap
# containing a dilute gas, and ``C`` is a repulsion constant.  This
# library follows the normalisation convention ``\int |\psi|^2=1``,
# so the number of particles should be absorbed into ``C``.
# 
# For numerical purposes, the factors of 2 are an unsettled matter
# of taste.  Therefore the package provides a parameter `:hbm` for
# the user to set to ``\hbar/m`` in the unit scheme of their choice.

using Superfluids
Superfluids.default!(:hbm, 1)

# When ``\hbar/m`` is made dimensionless, time is identified with
# area, according to the quadratic dispersion relation for de Broglie
# waves.
# 
# In the case that ``V={1\over 2}m\omega x^2`` is a quadratic
# potential, a common convention is to define the time unit by
# ``\omega=1``, and use the characteristic trap length
# ``\sqrt{\hbar/m\omega}`` as a length unit.  With the setting `:hbm
# = 1`, which is the default and used in this manual, the GPE becomes
# ```math
# i\psi_t = {1\over2}\left(-\nabla^2 + x^2 +
#     {C\over\hbar}\left({m\omega\over\hbar}\right)^{N/2}|\psi|^2\right)\psi.
# ```
# Where ``N`` is the dimension of the space on which the GPE is
# being solved, and the repulsion constant is now dimensionless.
# 
# Steady states of the order parameter satisfy a static GPE,
# ```math
# {\cal L}\psi = {\mu\over\hbar}\psi
# ```
# 
# Around a steady state ``\psi_0``, small excitations have the form
# ```math
# \psi(t,x) = \psi_0(x) + \alpha u(x)e^{i\omega t} + \alpha^\ast v^\ast(x)e^{-i\omega t},
# ```
# where ``u`` and ``v`` satisfy the Bogoluibov-de Gennes eigenproblem (TODO check conjugates)
# ```math
# ({\cal L}+C|\psi_0|^2) u + C\psi^2v = \omega u
# ```
# ```math
# ({\cal L}+C|\psi_0|^2) v + C\psi^{2\ast}u = -\omega v
# ```
# 
# 
# ## Superfluids
# 
# Optional (ψ₀, discretisation) pair
# 
# hbm initialised from default
# 
# Potential can be a function, or a (V, discretisation) pair
# 
# ## Rotating frames
# 
# A `Superfluid` can have a rotation rate, but this can be overridden by the `Ω` keyword argument.
# 
# ## Discretisations
# 
# The `Superfluids` package represents order parameters, and other
# fields, as the standard Julia `Array` type.  The abstract type
# `Discretisation{N}`, and its subtypes `FDDiscretisation{N}` and
# `FourierDiscretisation{N}`, map these arrays to continuous fields.
# The discretisation interface is specified in the developer manual.
#
# The following defines a 2nd order finite-difference discretisation, on a 10×10 grid with step 0.3 along each axis

d = FDDiscretisation{2}(10, 0.3)

# Julia functions are discretised as follows

V(x,y) = x^2 + y^2
u = sample(V,d)
ψ(z) = z/√(1+abs2(z))
q = argand(ψ,d)

# and a plot recipe displays them as complex phase portraits, taking
# a discretisation where the axes would usually go

plot(d,q)

# Currently, only scalar fields are implemented.
# 
# TODO Docstrings for `FDDiscretisation` and `FourierDiscretisation`
# 
# For all types of discretisation that allow it, the elements of `u`
# and `q` are multiplied by their quadrature weights, so that `sum(u)`
# is the integral of `V`.  Normalised vectors are very widely supported
# in numerical analysis, so a great deal of glue code can be avoided
# by representing normalised functions as normalised arrays.
# 
# The inverse of `sample` is `interpolate`.

V1 = interpolate(u, d)
V(0.2,0.3) ≈ V1(0.2,0.3) ≈ interpolate(u, d, 0.2, 0.3)

# ### Operators
# 
# 
# Most physicists are familiar with Schrödinger's equation.  This is
# linear in the wave function ψ, so the normalisation of ψ can be
# arbitrary: if ψ is a solution, so is cψ for any scalar c.  The
# Gross-Pitaevskii equation, on the other hand, includes a nonlinear
# term g|ψ|²ψ.  Therefore, the normalisation of ψ is not arbitrary.
# Physically, this corresponds to the wave function describing a
# single particle, while the order parameter describes an arbitrary
# number, and its norm corresponds to the number of particles.
# 
# There is however some arbitrariness, because of the repulsion
# constant g.  If the norm of ψ varies as 1/√g, the term g|ψ|² stays
# constant.  So, for example, for N particles you can have |ψ|²=N and
# g a purely atomic constant, or |ψ|²=1 and a constant gN.  Any
# numerical package needs to set a convention.  The package
# `ColdAtomConventions` allows things to be constructed using other
# conventions, and the results translated back.
# 
# In this package, the convention is that arrays representing order
# parameters have unit 2-norm.  This is very convenient numerically.
# This causes a subtlety, because the values stored in the array
# depend on how the x axis is discretised, as well as the function
# ψ(x).  For example, if the number of grid points is doubled, the
# array values must be divided by √2 to maintain the norm.  This isn't
# an issue for the linear derivative and poential operators.  The
# implementation section below explains how the |ψ|² operator allows
# for it.
# 
# In future, this package will include more complicated discretisations,
# not just Fourier ones with equispaced grids.  For these discretisations,
# the arrays for wave functions will still have unit norm.  To allow
# this, they will contain values of ψ multiplied by the quadrature
# weights for that discretisation.  The operator that multiplies by
# |ψ|² will allow for that, in the same way as it allows for grid
# size now.
# 
# 
# ### Interpolation
# 
# Interpolating between discretisations
# 
# Interpolating time series between Superfluids with different hbm
# 
# Plotting automatically interpolates
# 
# ## Steady state relaxation
# 

The basic relaxation routine is `steady_state`.  This takes a frame rotation frequency, and a list of vortex positions, and relaxes the continuous part of the order parameter to a minimum energy given those constraints.  Vortices can be specified with `(position, winding_number)` pairs, or just by a position, in which case the winding number is 1.  Positions are real pairs or complex numbers.

# 
# 
# A key feature of `Superfluids` is the ability to relax the density and phase of the order parameter, while constraining the positions of the vortices it contains, and other parameters such as the frame rotation rate.  This is useful in several contexts.  The first is that many vortex lattices are stationary, but energetically unstable—the stationary point is a saddle point in energy space, not a minimum.  (It can't be a maximum, because there are lots of ways to add energy.)  The simplest case is a single precessing vortex, in systems whose ground state is either vortex-free, or a central vortex, depending on the frame rotation rate.  Many multi-vortex lattices have stable and unstable regions.  E.g., pair of vortices is stable near the trap centre, and unstable near the edges, with a transition at roughly half the Thomas-Fermi radius.  To compute the unstable stationary states, it is necessary to relax an order parameter that still has vortices, while adjusting their positions and the rotation rate to find the stationary point.
# 
# The other issue is that, when the ground state is a lattice, there are usually several stationary vortex configurations with very similar energies.  Unconstrained relaxation will give an order parameter with one of these configurations, but not necessarily the one whose energy is a global minimum.  With constrained relaxation, it is possible to find the minimum energy state with every plausible configuration, and compare them to find the global minimum.  As a bonus, the global minimum is more stable, while unconstrained relaxation can jump between the local minima with small changes in the system parameters.
# 
# A concern with this approach is that the constraint distorts the order parameter, so it does not relax to the true ground state.  Sometimes, that is inevitable.  If the rotation rate and the vortex position have been chosen to be inconsistent with each other, then the system does not have a stationary state that satisfies the constraints.  Any state that is constrained to be stationary must be distorted somehow, and the form of the distortion is somewhat arbitrary.  In Superfluids, small values of the parameter `a` will cause the distortion to be localised near the vortex, so that the fluid away from the vortex is locally in a true stationary state.  Large values will spread the distortion out across a large part of the fluid, which often looks more physically reasonable.

# On the other hand, suppose constraints have been chosen for which the system does have a stationary state (up to numerical precision in the constraints).  In the stationary state, the vortex positions are stationary too.  Therefore, as the system approaches its constrained equilibrium state, the effect of the projection that makes it satisfy the constraint reduces.  When the system reaches its constrained relaxed state, the constraint has no effect at all, and the relaxation process becomes exactly the same as the unconstrained one.  So, in the case where the unconstrained system has a stationary state that satisfies the constraintes, the constrained relaxation algorithm will find it.

# There are two apparent ways to improve the vortex constraints.  The first is to use parallel transport, instead of doing a relaxation step and then projecting back on to the manifold.  Parallel transport is a bit strange.  The order parameters with a vortex at a given position form a linear submanifold: if you add two vortices, you get a vortex.  So, once the descent direction is projected to have a vortex at the right place, any order parameter along the line of descent should satisfy the constraints.  What goes wrong?

Suppose there is a vortex at the origin, and the energy decreases if it moves in the positive x direction.  The descent direction is ∂/∂a (z-a) = -1, at least inside the vortex core.  Projecting this gives -1+δ(z), where δ(z) is a band-limited delta function.  So most of the core moves as if there were no constaint, but a delta spike is added to warp the phase singularity within a single pixel.  As we go along the descent direction, the vortex core actually does move, in the unconstrained way, but then a hole is poked at the constrained vortex position.  That's exactly what happens in simulations!

What we really want to do, is move along the descent direction, while keeping the vortex in the same place.  This is no longer a linear projection operator.  The further the vortex moves, the larger a Gaussian needs to be added, so that the singularity is shifted back where it should be, but the order parameter remains fairly linear around the singularity.

So we don't just want a descent line.  It needs to be a descent curve, where the further we go, the wider a Gaussian is added to return the vortex to its original position.  There is a way to estimate how large a Gaussian is needed, based on the coefficient of (z-a) in the order parameter around the vortex.  It isn't clear how to do that within the Optim framework, but it would be interesting to extend the framework to allow it.

This also give an idea for dealing with vortices entering from the edge of the condensate.  You need to add a wide enough Gaussian to push them back to the edge.  Provided a wide enough Gaussian is added at each vortex core, there is no way for a vortex to form in the order parameter.  Adding sums of Gaussians shouldn't give a phase singularity.  Or if it does, that gives a limit on how far to follow the line of descent.


# 

# # 
# ## Vortices and solitons
# 
# In 1D, soliton locations are specified as a vector of real numbers.
# 
# In 2D, vortex core locations are a vector of pairs or complex numbers.
# 
# TODO figure out how to specify vortex lines in 3D (and soliton lines in 2D, if they exist)
# 
# As well as unconstrained relaxation, the `steady_state` function
# supports relaxing the continuous part of the order parameter, while
# maintaining a lattice of vortices with given positions.  For example,
# a central vortex can be obtained (in 2D) by

ψ = steady_state(rvs=[0])

# In this case, `ψ` actually is a stationary state.  This can be
# confimed by

ψ ≈ steady_state(initial=ψ)

# This is a bit of a fluke: in general, `steady_state` could return
# `u*ψ`, where `u` is any complex number with `abs(u)=1`.  A more
# reliable test is to project out the orthogonal component

let q = steady_state(initial=ψ)
    norm(q - dot(ψ,q)*ψ/norm(ψ)^2)

# Not every lattice of vortices is stationary.  For example, in the
# laboratory frame, an offset vortex precesses in an orbit.  In this
# case, `steady_state` returns the least energy lattice with vortices
# at those positions.

steady_state(rvs=[-1, 1])

# In a frame that rotates at the precession frequency, these lattices
# will be stationary

steady_state(rvs=[1], Ω=...)

# The `rvs` argument ensures that the specified vortices are present.
# The current implementation does not ensure that other vortices are
# absent.  This can fail in two ways.  If the frame rotation rate is
# too slow compared to the lattice precession rate, anti-vortices
# will form right next to some of the vortices.  This produces an
# odd-looking wave function, with the hole for a vortex core, but no
# phase winding around it.  (This is not a physically meaninful state:
# the least energy problem becomes highly contrived when the vortices
# are constrained to form a non-stationary lattice.)

steady_state(rvs=[3], Ω=0)

# Conversely, if the frame rotation rate is too high, the result will
# have vortices in the places specified, but the relaxation algorithm
# will add some extra ones to reduce the energy.

steady_state(rvs=[-1,1], Ω=0.8)

# There is a numerical parameter `as`, that can amelorate the missing
# vortices to some extent.  See the implementaion section for a
# description of how this works.
# 
# A vortex is specified either as either a core location `rv`, or a
# two-element collection `(rv, n)`.  The location `rv` is specified
# in any of the acceptable forms of coordinates, and `n` is a winding
# number, which defaults to 1 when only `rv` is supplied.
# 
# Relaxing vortex positions with fixed rotation frequency
# 
# Finding a frequency where a set lattice is steady
# 
# ## GPE dynamics
# 

# The Gross-Pitaevskii equation is solved by the function `integrate`.  This takes an initial order parameter, and a vector of times to return solutions.  It returns a vector of solutions.

# The solver uses Julia's `DifferentialEquations` package.  A Version 1 goal is to implement RK4IP, the standard GPE integration method, or to provide matrix exponentials for the ODE methods.

# 
# Plotting dynamics gives an animation: `plot(::Discretisation{N}, ts, ψs)'.  Where `ψs` can be an `N+1` array, or a vector of `N` arrays. 
# 
# ## Bogoliubov de-Gennes modes
# 
# 
# When the Bogoliubov de-Gennes equations are diagonalised, they give
# a symplectic(?) matrix, not a Hermitian one.  That is to say, the
# eigenvectors are orthogonal under an inner product, but it is the
# Lorentz one instead of the Euclidean one: the |u|² and |v|² components
# are subtracted rather than added.  The science of diagonalizing
# such matrices is poorly developed.  Therefore, this package focuses
# on generating the matrix (or functions that implement the linear
# transformation, for iterative methods), and on validating and
# cleaning up the output.  It largely leaves the diagonalisation
# routine up to the user.  The built in eigen works fine for small
# problems.
# 
# The functions bdg_matrix and bdg_operators take a system, a
# discretisation, and a condensate order parameter.  They return the
# discretised BdG matrix.  This gets large quickly!  An n×n grid has
# n² points, so a (u,v) vector has 2n² elements, and the BdG matrix
# over these vectors has 4n⁴ elements.
# 
# The eigenproblem has an inherent degeneracy.  For any solution
# (u,v,ω), there is also a solution (v,u,-ω) check conjugates.  These
# represent the same physical mode.  The usual approach, and the
# correct one from a quantum point of view, is to keep the positive
# norm modes, where |u|²>|v|².  The function bdg_output checks that
# the modes are paired in the expected way, and returns the positive
# norm ones.  It also reshapes the matrix of eigenvectors into arrays
# of u and v matrices.  There is a raw option, which does not perform
# the checks, but does the reshaping.  This is useful when odd things
# happen with the diagonalisation—as they often do, because the usual
# routines aren't designed for symplectic problems.
# 
# The non-orthogonal eigenvectors interact in inconvenient ways with
# degeneracies.  People have found that the diagonalisation is more
# stable numerically if the condensate mode is removed.  (It's a zero
# mode, so other modes must be orthogonal to it in the Euclidean
# sense.)  The bdg_matrix routine has an option to project into the
# space orthogonal to the condensate, which is done by Householder
# reflection.
# 
# David Hutchinson came up with a better way of doing the diagonalisation,
# for the special case where the order parameter is real.  I haven't
# implemented this yet, because the library has largely been used for
# vortex lattices, but it's the to-do list.

# The matrix functions
# 
# Normalisation with `norm2(u)-norm2(v) == 1`.
# 
# Plotting a mode set to give J vs ω
# 
# Animating modes
# 

## Future Directions

# Currently, `System` and `Discretisation` need to be passed around separately to the arrays of wave functions.  It would be good if Julia had an effective way to attach annotations like these to arrays, especially when the arrays are saved to JLD files.  The drawback to that is that array types nest, and there is currently no way for the compiler to figure out that, somewhere inside all the annotation types, there are `Diagonal` arrays with optimised methods.  Instead, it tends to index into the arrays, make a dense copy of their elements, and use the GE routines.

# Solving this will require some changes to the Julia array ecosystem.  There should be a conventional type nesting hierarchy, with structured storage at the bottom (Diagonal), lazy restrides above that (reshape, transpose, etc), then annotations.  The upper layers could also be rationalised.  Almost all of the lazy reshape types could be absorbed into one `RestridedArray`, with a type parameter for conjugation.  All the annotation types could be type parameters on an `AnnotatedArray{Union of annotation types}`, although that might require a special `Annotation{T,R,S,...}` to be added to Julia's type system, to avoid everything bcoming an `AnnotatedArray{Any}`.  The idea would be for compatible annotations to be promoted and merged.

Inside that idea is a simpler broadcasting interface, where the container types are promoted by a similar mechanism to the element types.

# ## Parameters and defaults
# 
# Like `Plots.jl`, this package has a lot of parameters that users will
# sometimes need to adjust. Moreover, almost every function takes a
# `Superfluid` or a `Discretisation` argument, or both, and it is common
# for these to be the same for every call in a simulation. Configurable
# defaults are provided to suppress the redundant arguments in
# user-facing functions, through the `default` and `default!` functions
# that are used throughout this manual.
# 
# Most exported functions take a `Superfluid` or a `Discretisation` as
# their first argument, or both in that order as their first two
# arguments. The exceptions are `sample`, `argand` and `steady_lattice`,
# which put their function argument first to allow the use of `do`
# syntax. Once `Superfluids.default!` has been called with an argument
# of type `Superfluid` or `Discretisation`, these arguments can be
# omitted, and the default will be used.
# 
# The remaining defaults are keyword arguments to `Superfluid`
# functions, most of which are passed on to `Plots`,
# `DifferentialEquations` and `Optim`. Those arguments are described in
# the relevant sections of this manual, along with their default values
# when `Superfluids` is loaded. The syntax to override the default is
# `default!(:parameter, value)`. The default superfluid can be set with
# `default!(:superfluid, s)`, and the discretisation with
# `default!(:discretisation, d)`.
# 
# # ```@docs
# default
# default!
# ```
# 
# # Manual
#
# Copy the format of `Base` and `Documenter`.
# 
# # Internals
# 
# ## The discretisation interface
# 
# A discretised wave function is an array.  A normalised wave function should correspond to a normalised array, i.e. absorb the quadrature weights into the coefficients.
# 
# `primitive_operators`
# 
# `dif`
# `sample`
# 
# TODO how do vortices work with general discretisatons?  Presumably a function that takes `rv`, and returns the dual vector q -> ψ(rv).
# 
# # ## How the vortex detector works
# 
# ## How the vortex pinning works.  General idea: ψ(r_v) = 0 + ∇(r_v)[dψ]
# + O(dψ²).  Project dψ onto the null space of J.  Retraction is a
# multivariate optimization in concept, but one step in direction
# -∇(rv) works in practice.
# 
# Structured BdG matrices

## Solving the BdG eigenproblem

The current implementation of BdG solving is brute force.  The
elements of the BdG operator are written to a dense matrix, which
is diagonalised with the LAPACK eig routine.  There is a complication:
there are often several zero modes, which form a degenerate eigenspace.
When a hermitian matrix is diagonalised, that would not be a problem,
because the solver would return a set of orthogonal vectors spanning
the space.  However, the BdG matrix is not hermitian, so the solver
will return a quite arbitrary set of vectors spannign the space.
To fix this, the matrix `B` is transformed to `H* B H`, where `H`
is a Householder reflection matrix, whose columns are a basis for
the space orthogonal to the condensate mode.  Then `eig` gives all
the modes except for the condensate mode, which is added on explicitly
by `bdg_output`.

It would be (very) nice to solve this by an iterative method.  The
problem is the degeneracy, where every frequency occurs as ω and
-ω.  Iterative solvers pick off the eigenvalues at the edge of the
spectrum, and we're usually interested in the modes with small ω,
which line in the middle of the degenerate spectrum.  There is a
standard solution to this: instead of diagonalising `B`, diagonalise
`B⁻¹`.  However, this relies on a method of computing `B⁻¹`.  There
are iterative methods to do that, and no doubt someone has designed
an algorithm that combines iterative diagonalisation with shift and
iterative inversion.  If you know where to find that, please tell
me—my (half hearted) attempts to reinvent it have been numerically
unstable.

# 
# ### Normalization
# 
# * Arrays that represent wave functions are normalised with ``l^2``.
# 
# * Arrays that represent energy-valued functions are samples.
# 
# * Wave functions returned by interpolation are ``L^2`` normalized.
# 
# * The tricky bit is ``|\psi|^2`` in ``U(\psi)``, which must be adjusted as if the ``L^2`` norm were 1.
# 
# # Roadmap
# 
# ## Version 1
# 
# * Support FD and Fourier spectral discretisation
# 
# * 1D and 2D are feature complete and documented
# 
# * 3D with vortices is underway
#
# * Multiple components are documented
# 
# * 100% test coverage
# 
# ## Version 2
# 
# * 3D and multiple components are fully implemented
#
# * Support for Wigner sampling initial states
# 
# * Julia has a ':LM' iterative solver, and the necessary sparse matrix inverses, for the BdG eigenproblem
# 
# * Efficient support for `DiffEqOperators`
# 
# ## Future
# 
# * Green's function preconditioning
# 
# * Dynamics can be solved by GPU
# 
# * Fornberg spectral discretisation
# 
# * Manage projection and aliasing for phase-space Monte-Carlo
