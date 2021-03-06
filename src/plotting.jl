# Plot recipes

using RecipesBase, ComplexPhasePortrait, PlotUtils

@recipe function f(d::Sampling{2}, u::Matrix{Complex{Float64}})
    x, y = daxes(d)
    let xs = default(:xlims), ys = default(:ylims)
        xlims --> (isnothing(xs) ? (x[1], x[end]) : xs)
        ylims --> (isnothing(ys) ? (y[1], y[end]) : ys)
    end
    yflip := false
    aspect_ratio --> 1
    tick_direction --> :out
    x[:], y[:], transpose(saneportrait(u) .* abs2.(u) / maximum(abs2, u))
end

@recipe function f(d::Sampling{2}, u::Matrix{Float64})
    x, y = daxes(d)
    let xs = default(:xlims), ys = default(:ylims)
        xlims --> (isnothing(xs) ? (x[1], x[end]) : xs)
        ylims --> (isnothing(ys) ? (y[1], y[end]) : ys)
    end
    yflip := false
    aspect_ratio --> 1
    tick_direction --> :out
    if isnothing(default(:clim))
        x[:], y[:], transpose(sense_portrait(u))
    else 
        x[:], y[:], transpose(sense_portrait(u, default(:clim)))
    end
end

@recipe f(d::Discretisation{2}, u::Matrix{Float64}) = (d, Complex.(u))

saneportrait(u) = reverse(portrait(u), dims = 1)

"ComplexPhasePortrait, but with real sign instead of phase"
function sense_portrait(xs, mag=maximum(abs, xs))
    C = cgrad([:cyan, :white, :red])
    # TODO stability at |x| ≈ mag
    [C[iszero(mag) ? 1/2 : (x+mag)/2mag] for x in xs]
end

# TODO plot recipes for animations

# BdG spectra

@userplot BdGSpectrum

@recipe function f(uvs::BdGSpectrum)
    # TODO don't require s for J operator
    s, d, ws, us, vs = uvs.args
    J = operators(s, d, :J) |> only
    js = [dot(u, J(u)) + dot(v, J(v)) |> real for (u, v) in zip(us, vs)]
    seriestype := :scatter
    js, ws
end
