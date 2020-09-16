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

@testset "Compare operators to explicit matrices" begin
    d = FDDiscretisation{2}(10, 10/9)
    s = Superfluid{2}(500, (x,y) -> (x^2+y^2)/2, hbm=rand())
    Ω = rand()
    # TODO fuzerate wave functions
    ψ = cloud(d)
    es = [reshape(e, d.n, d.n)
        for e = eachcol(Matrix{Complex{Float64}}(I, d.n^2, d.n^2))]
    
    function compare(a, f)
        A = Superfluids.operators(s,d,a) |> only
        M = Superfluids.matrices(s,d,Ω,ψ,a) |> only
        @test Superfluids.expand(q->f(A,q),es) ≈ M
    end
    
    compare(:U, (U,q)->U(q,abs2.(ψ)))
    compare(:L, (L,q)->L(q,abs2.(ψ);Ω))
    compare(:H, (H,q)->H(q,abs2.(ψ);Ω))
    compare(:T, (T,q)->T(q))
    compare(:V, (V,q)->V(q))
    compare(:J, (J,q)->J(q))
end

@testset "Compare Hartree modes to matrix eigenvectors" begin
    hbm = rand()
    Ω = 0.01*hbm	# break degeneracy
    s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2; hbm)
    d = FDDiscretisation{2}(40, 15/39)
    L, J = operators(s,d,:L,:J)
    ψ = steady_state(s,d;Ω)
    μ = dot(L(ψ;Ω), ψ) |> real
    M = matrices(s,d,Ω,ψ,:L) |> only
    ew, ev = eigen(M)
    ws, us = hartree_modes(s,d,ψ,10,Ω)
    @test norm(imag(ws), Inf) < 1e-10
    ws = real(ws)
    @test ws[1] ≈ μ
    @test ws ≈ ew[eachindex(ws)]
    for (j, u) = pairs(us)
        @test abs(dot(u, ev[:,j])) ≈ 1
    end
    jj = [dot(u,J(u)) for u in us]
    @test norm(imag(jj), Inf) < 1e-10
    @test norm(round.(jj) - jj, Inf) < 0.1
end

@testset "Compare BdG modes with matrix" begin
    # leave hbm = 1 to keep core smaller than trap
    Ω = 0.01
    s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2, hbm=0.5)
    d = FDDiscretisation{2}(40, 15/39)
    L, H, J = Superfluids.operators(s,d,:L,:H,:J)
    ψ = steady_state(s,d;Ω)
    @test norm(ψ) ≈ 1
    
    # Check that this problem is solvable
    M = Superfluids.BdGmatrix(s,d,Ω,ψ)
    ew, ev = eigs(M, nev=20, which=:SM)
    @test norm(imag(ew), Inf) < 1e-3
    ew = real(ew)
    @test ew[1] ≈ 0 atol=1e-6
    for j = 2:2:length(ew)
        @test ew[j-1] ≈ -ew[j] rtol=1e-3
    end
    uu = [reshape(u[1:d.n^2],d.n,d.n) for u in eachcol(ev)]
    vv = [reshape(v[d.n^2+1:end],d.n,d.n) for v in eachcol(ev)]
    @test norm(uu[1]) ≈ 1/√2
    @test norm(vv[1]) ≈ 1/√2
    @test abs(dot(uu[1],ψ)) ≈ 1/√2 rtol=1e-3
    @test abs(dot(vv[1],ψ)) ≈ 1/√2 rtol=1e-3
    ix = ew .≥ 0
    ew = ew[ix]
    uu = uu[ix]
    vv = vv[ix]
    
    # Check expansions
    hw, hv = hartree_modes(s,d,ψ,100,Ω)
    @test norm(imag(hw), Inf) < 1e-6
    hw = real(hw)
    
    uns = [norm(u)^2 for u in uu]
    ups = [sum(abs2, dot(h,u) for h in hv) for u in uu]
    @test uns ≈ ups rtol=0.01
    vns = [norm(v)^2 for v in vv]
    vps = [sum(abs2, dot(conj(h),v) for h in hv) for v in vv]
    @test vns ≈ vps rtol=0.01
    
    ws, us, vs = bdg_modes(s, d, ψ, Ω, 100)
    @test norm(imag(ws), Inf) < 1e-3
    ws = real(ws)
    @test ew ≈ ws[eachindex(ew)] rtol=0.05
    for j = eachindex(ew)
        u, v = us[j], vs[j]
        u0, v0 = uu[j], vv[j]
        @test abs(dot([u0[:]; v0[:]], [u[:]; v[:]])) ≈ 1 rtol=0.03
    end
end

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

@testset "BdG matrix annihilates zero modes"  begin
    # ew, us = hartree_modes(s, d, ψ, nev, Ω);
    # Umat = hcat([u[:] for u in us]...);
    # cs = Umat'*q[:];
    # ds = conj(cs)
    # M = Superfluids.BdGmatrix(s, d, Ω, ψ, us)
    # A = M[1:10, 1:10];
    # B = M[1:10, 11:20];
    # C = M[11:20, 1:10];
    # D = M[11:20, 11:20];
    # Mu = Umat*(A*cs+B*ds);
    # Mu = reshape(Mu, d.n, d.n);
    # Mv = conj(Umat)*(C*cs+D*ds);
    # Mv = reshape(Mv, d.n, d.n);
end