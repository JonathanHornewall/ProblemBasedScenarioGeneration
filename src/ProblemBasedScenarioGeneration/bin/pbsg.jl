#!/usr/bin/env julia
#
# ProblemBasedScenarioGeneration CLI
#
# Usage:
#   julia bin/pbsg.jl train -p newsvendor --epochs 10 -v -o model.jls
#   julia bin/pbsg.jl info -c model.jls
#   julia bin/pbsg.jl test -c model.jls --n-test 50
#   julia bin/pbsg.jl continue -c model.jls --epochs 5 -o model_v2.jls
#
# Run from the package root:
#   julia --project=. bin/pbsg.jl <command> [options]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ProblemBasedScenarioGeneration
ProblemBasedScenarioGeneration.cli_main()
