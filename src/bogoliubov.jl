# Bogoliubov-de Gennes problems

"""
    ωs, us, vs = modes(s, d, ψ, Ω, nev)
    
Return vectors of the lowest energy BdG modes

Normalised with ∥u∥² - ∥v∥² = 1.
"""
function modes(s, d, ψ, Ω, nev)
    N = d.n
    B = BdGmatrix(s,d, Ω, ψ)
    ωs,uvs = Arpack.eigs(B; nev=2*nev, which=:SM)
    @info "max imag frequency" iw=maximum(imag, ωs)
    ωs = real(ωs)
    ixs = findall(ωs .> 0)
    ωs[ixs],
        [reshape(uvs[1:N^2, j], N, N) for j = ixs],
        [reshape(uvs[N^2+1:end, j], N, N) for j = ixs]
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

"""
    ωs, us = hartree_modes(s, d, ψ, n)

Return the lowest `n` hartree eigenfunctions
"""
function hartree_modes end

@defaults function hartree_modes(s, d, ψ, n, Ω=0.0)
    L = operators(s, d, :L) |> only
    Lop = LinearMap{Complex{Float64}}(d.n^2) do u
        u = reshape(u, d.n, d.n)
        L(ψ,u;Ω)[:]
    end
end

# function BdGmatrix(s, d, Ω, ψ)
#     # TODO make this a LinearMap
#     apop(f) = x->f(reshape(x, d.n, d.n))[:]
#     t, v, w, j, l = operators(s, d, :T, :V, :U, :J, :L)
#     u(q) = w(ψ,q)
#     qq(q) = @. ψ^2*q/d.h^2
#     pp(q) = @. conj(ψ)^2*q/d.h^2
#     
#     μ = UniformScaling(real(dot(ψ, L(ψ))))
#     T, V, J, U, QQ, PP = [
#         LinearMap{Complex{Float64}}(apop(A), d.n^2) for
#             A = [t, v, j, u, qq, pp]
#     ]
#         
#     [
#         T+V+2U-μ-Ω*J   -s.C*QQ;
#         s.C*PP    -T-V-2U+μ-Ω*J
#     ]
# end