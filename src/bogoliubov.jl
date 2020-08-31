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
    umode(j) = reshape(uvs[1:N^2, j], N, N)
    vmode(j) = reshape(uvs[N^2+1:end, j], N, N)
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
        T+V+2U-μ-Ω*J   -s.C/d.h*Q.^2;
        s.C/d.h*conj.(Q).^2    -T-V-2U+μ-Ω*J
    ]
end