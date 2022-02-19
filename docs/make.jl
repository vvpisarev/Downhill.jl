push!(LOAD_PATH,"../")

using Documenter
using Downhill

makedocs(
    ;
    sitename="Downhill.jl documentation",
    modules=[Downhill],
    pages = [
        "index.md",
        "Optimization Methods" => "core_types.md",
        "Basic Functions" => "functions.md",
        "Customization" => "wrappers.md"
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

deploydocs(
    repo = "github.com/vvpisarev/Downhill.jl.git",
)
