# Bogoliubov-de Gennes problems

"""
    ωs, us, vs = bdg_modes(s, d, ψ, Ω, nev)
    
Return vectors of the lowest energy BdG modes

Normalised with ∥u∥² - ∥v∥² = 1.
"""
function bdg_modes(s, d, ψ, Ω, nev)
    # TODO how many hartree modes to include?
    # write test to double nev, compare first half of modes
    _, us = hartree_modes(s, d, ψ, nev, Ω)
    U = hcat((u[:] for u in us)...)
    B = expand(bdg_operator(s,d,ψ,Ω), us, conj.(us))
    ew, ev = eigen(B)
    @info "max imag frequency" iw=maximum(imag, ew)
    ew = real(ew)
    us = [reshape(U*ev[1:nev, j], d.n, d.n) for j = eachindex(ew)]
    vs = [reshape(conj.(U)*ev[nev+1:end, j], d.n, d.n) for j = eachindex(ew)]
    ixs = findall(norm.(us) .> norm.(vs))
    @assert length(ixs) == nev
    ew[ixs], us[ixs], vs[ixs]
 end

"expand f(u) as a matrix over us"
function expand(f, us)
    [dot(us[j], f(us[k])) for j = eachindex(us), k = eachindex(us)]
end

"expand u', v' = f(u,v) as a matrix over us, vs"
function expand(f, us, vs)
    x(g,as,bs) = [dot(as[j], g(bs[k])) for j = eachindex(as), k = eachindex(bs)]
    [x(u->f(u,zero(u))[1], us, us)  x(v->f(zero(v),v)[1], us, vs);
    x(u->f(u,zero(u))[2], vs, us)  x(v->f(zero(v),v)[2], vs, vs)]
end

"""
    bdg_operator(s,d,ψ,Ω)

v is conjugated
"""
function bdg_operator(s,d,ψ,Ω)
    T, V, U, J, L = operators(s, d, :T, :V, :U, :J, :L)
    
    μ = dot(L(ψ;Ω), ψ) |> real
    (u, v) -> (
        T(u)+V(u)+2U(u,abs2.(ψ))-μ*u-Ω*J(u)-U(v,ψ.^2),
        U(u,conj(ψ).^2)-T(v)-V(v)-2U(v,abs2.(ψ))+μ*v-Ω*J(v)
    )
end

function BdGmatrix(s, d, Ω, ψ)
    # TODO make this a BlockBandedMatrix
    T, V, U, J, L = matrices(s, d, Ω, ψ, :T, :V, :U, :J, :L)
    Q = diagm(0=>ψ[:])
    μ = sum(conj(ψ[:]).*(L*ψ[:])) |> real |> UniformScaling
    [
        T+V+2U-μ-Ω*J   -s.C/d.h^2*Q.^2;
        s.C/d.h^2*conj.(Q).^2    -T-V-2U+μ-Ω*J
    ]
end

"Expand u over us, v* over us*"
function BdGmatrix(s, d, Ω, ψ, us)
    T!, V!, U!, J!, L = operators(s, d, :T!, :V!, :U!, :J!, :L)
    nev = length(us)
    M = Array{Complex{Float64}}(undef, 2nev, 2nev)
    μ = dot(ψ, L(ψ; Ω)) |> real
    norm(L(ψ; Ω)-μ*ψ) > 1e-3 &&
        @warn "Non-stationary order parameter" residual=norm(L(ψ; Ω)-μ*ψ)
    w = similar(us[1])
    for j = 1:nev, k = 1:j
        # Diagonal blocks
        u = us[k]
        w .= 0
        T!(w,u)
        V!(w,u)
        U!(w,2,ψ,u)
        a = dot(us[j],w)
        j == k && (a -= μ)
        
        w .= 0
        b = dot(us[j], J!(w,Ω,u))
        M[j,k] = a-b
        M[k,j] = conj(a-b)
        # conj(v) ⋅L(conj(u))) = conj(u ⋅L(v))
        M[k+nev, j+nev] = -a-b
        M[j+nev, k+nev] = conj(-a-b)
    end
    
    for j = 1:nev, k = 1:nev
        # Off-diagonal blocks
        M[j, k+nev] = -s.C/d.h^2*dot(us[j], @. ψ^2*us[k])
        M[j+nev, k] = s.C/d.h^2*dot(conj(us[j]), @. conj(ψ)^2*conj(us[k]))
    end
    
    M
end

"""
    ωs, us = hartree_modes(s, d, ψ, n, Ω)

Return the lowest `n` hartree eigenfunctions
"""
function hartree_modes end

function hartree_modes(s, d, ψ, nev, Ω)
    L = operators(s, d, :L) |> only
    Lop = LinearMap{Complex{Float64}}(d.n^2) do u
        u = reshape(u, d.n, d.n)
        L(u,abs2.(ψ);Ω)[:]
    end
    ws, us = eigs(Lop; nev, which=:SR)[1:2]
    ws, [reshape(u, d.n, d.n) for u = eachcol(us)]
end
