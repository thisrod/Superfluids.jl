# Plot recipes

using RecipesBase, ComplexPhasePortrait

@recipe function f(d::Discretisation{2}, u::Matrix{Complex{Float64}})
    x, y = d.xyz
    xlims --> (x[1], x[end])
    ylims --> (y[1], y[end])
    yflip := false
    aspect_ratio --> 1
    tick_direction --> :out
    x[:], y[:], transpose(saneportrait(u).*abs2.(u)/maximum(abs2,u))
end

@recipe f(d::Discretisation{2}, u::Matrix{Float64}) = (d, Complex.(u))

saneportrait(u) = reverse(portrait(u), dims=1)
