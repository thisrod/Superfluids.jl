using Literate, Documenter, Superfluids

Literate.markdown("src/index.jl", "./src", documenter=true)

makedocs(
    modules=[Superfluids],
    sitename="My Documentation"
)