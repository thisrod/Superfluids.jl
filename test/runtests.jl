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
#         @test D(u, 1, d, 2) ≈ -(π/l)^2*u
#         @test D(D(u, 1, d), 1, d) ≈ D(u, 1, d, 2)
#         @test D(u, 2, d, 2) ≈ -(π/l)^2*u
#         @test D(D(u, 2, d), 2, d) ≈ D(u, 2, d, 2)
    end
end

# TODO extend fuzerate with a "this is no less converged than the last time the test ran" macro and an order "this test data should converge better than that does"

@testset "SHO ground state expectation values" begin
    s = Superfluid{2}(0, (x,y) -> (x^2+y^2)/2)
    for d = fuzerate(Discretisation)
        T, V, U, J = operators(s,d,:T,:V,:U,:J)
        z = argand(d)
        w = similar(z)
        w .= normalize(@. exp(-abs2(z)/2))
        @test dot(w,T(w)) ≈ 0.5 atol=0.02
        @test dot(w,V(w)) ≈ 0.5 atol=0.02
        @test dot(w,J(w)) ≈ 0.0 atol=1e-10
        @test dot(w,U(w)) ≈ 0.0
    end
end

@testset "SHO ground state is an eigenstate" begin
    s = Superfluid{2}(0, (x,y) -> (x^2+y^2)/2)
    for d = fuzerate(Discretisation)
        L = operators(s,d,:L) |> only
        z = argand(d)
        w = similar(z)
        w .= normalize(@. exp(-abs2(z)/2))
        @test dot(w,L(w)) ≈ 1.0 atol=0.02
        @test norm(L(w)-w) < 0.05
    end
end

@testset "chemical potential" begin
    # TODO check this analytically
    s = Superfluid{2}(500, (x,y) -> (x^2+y^2)/2)
    for d = fuzerate(Discretisation)
        L = operators(s,d,:L) |> only
        q = steady_state(s,d)
        @test dot(q, L(q)) ≈ 12.678 atol=0.02
    end
end

@testset "Hartree BdG matrix compared to explicit form"

end

@testset "Order parameter is zero mode" begin

end
