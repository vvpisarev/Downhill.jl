push!(LOAD_PATH,"../")

using Documenter
using DescentMethods

makedocs(
    ;
    sitename="Documentation",
    modules=[DescentMethods],
    pages = [
        "index.md",
        "Optimization Methods" => "core_types.md",
        "Basic Functions" => "functions.md",
    ]
)
