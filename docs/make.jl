import Pkg;

using Documenter
using SMCForecast

makedocs(
    sitename = "SMCForecast",
    format = Documenter.HTML(),
    modules = [SMCForecast],
    pages = ["index.md"]
)

deploydocs(;
    repo="https://github.com/SupplyChef/SMCForecast.jl",
    devbranch = "main"
)