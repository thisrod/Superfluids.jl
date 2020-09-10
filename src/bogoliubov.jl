# Bogoliubov-de Gennes problems

"""
    ωs, us, vs = modes(s, d, ψ, Ω, nev)
    
Return vectors of the lowest energy BdG modes

Normalised with ∥u∥² - ∥v∥² = 1.
"""
function modes(s, d, ψ, Ω, nev)
    # TODO how many hartree modes to include?
    _, us = hartree_modes(s, d, ψ, nev, Ω)
    B = BdGmatrix(s, d, Ω, ψ, us)
    ew, ev = eigen(B)
    @info "max imag frequency" iw=maximum(imag, ew)
    ew = real(ew)
    ixs = findall(ew .> 0)
    @assert length(ixs) == length(us)
    U = hcat((u[:] for u in us)...)
    ew[ixs],
        [reshape(U*ev[1:nev, j], d.n, d.n) for j = ixs],
        [reshape(U*ev[nev+1:end, j], d.n, d.n) for j = ixs]
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

function BdGmatrix(s, d, Ω, ψ, us)
    T!, V!, U!, J!, L = operators(s, d, :T!, :V!, :U!, :J!, :L)
    nev = length(us)
    M = Array{Complex{Float64}}(undef, 2nev, 2nev)
    μ = dot(ψ, L(ψ; Ω)) |> real
    norm(L(ψ; Ω)-μ*ψ) > 1e-3 &&
        @warn "Non-stationary order parameter" residual=norm(L(ψ; Ω)-μ*ψ)
    y = similar(us[1])
    for j = 1:nev, k = 1:j
        # Diagonal blocks
        u = us[k]
        y .= 0
        T!(y,u)
        V!(y,u)
        U!(y,2,ψ,u)
        a = dot(us[j],y)
        j == k && (a -= μ)
        y .= 0
        b = dot(us[j], J!(y,Ω,u))
        M[j,k] = a-b
        M[k,j] = conj(M[j,k])
        M[j+nev, k+nev] = -a-b
        M[k+nev, j+nev] = conj(M[j+nev, k+nev])
    end
    
    for j = 1:nev, k = 1:nev
        # Off-diagonal blocks
        M[j, k+nev] = -s.C/d.h^2*dot(us[j], @. ψ^2*us[k])
        M[j+nev, k] = s.C/d.h^2*dot(us[j], @. conj(ψ)^2*us[k])
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
        L(ψ,u;Ω)[:]
    end
    ws, us = eigs(Lop; nev, which=:SR)[1:2]
    ws, [reshape(u, d.n, d.n) for u = eachcol(us)]
end
