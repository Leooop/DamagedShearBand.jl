using DamagedShearBand
using Documenter

makedocs(;
    modules=[DamagedShearBand],
    authors="Léo Petit <leo.petit@live.fr> and contributors",
    repo="https://github.com/Leooop/DamagedShearBand.jl/blob/{commit}{path}#L{line}",
    sitename="DamagedShearBand.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Leooop.github.io/DamagedShearBand.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Leooop/DamagedShearBand.jl",
)
