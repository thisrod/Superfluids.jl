using Test

include("framework.jl")



"""
Consistency checks for operators and matrices:

julia> TT ≈ op2mat(T)

julia> VV ≈ diagm(0=>V[:])

julia> UU ≈ diagm(0=>U(ψ)[:])

julia> JJ ≈ op2mat(J)

function op2mat(f)
    M = similar(z,length(z),length(z))
    u = similar(z)
    for j = eachindex(u)
        u .= 0
        u[j] = 1
        M[:,j] = f(u)[:]
    end
    M
end

Check zero mode for BdGmat
"""

# Check that relaxed orbiting vortices and lattices have the right
# winding number around the trap edge.

# Check consistency when hbm, C and grid are changed invariantly
# Given one set, fuzerate identical ones

@testset "Derivatives of cloud" begin
    # Cloud is cos, so the second derivative flips the sign
    for d = fuzerate(Discretisation)
        l = d.h*(d.n+1)
        u = cloud(d)
        @test D(u, 1, d, 2) ≈ -(π/l)^2*u
        @test D(D(u, 1, d), 1, d) ≈ D(u, 1, d, 2)
        @test D(u, 2, d, 2) ≈ -(π/l)^2*u
        @test D(D(u, 2, d), 2, d) ≈ D(u, 2, d, 2)
    end
end

@testset "Primitive operators" begin

end