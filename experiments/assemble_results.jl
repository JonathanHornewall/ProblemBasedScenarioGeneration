#!/usr/bin/env julia
"""
Assemble all batch summary.csv files into a single grid_search_results.csv.

Usage:
    julia assemble_results.jl
"""

const EXPERIMENTS_DIR = @__DIR__
const PROBLEMS = ["newsvendor", "resource_allocation", "shipment_planning"]
const BATCHES  = ["batch_a", "batch_b"]
const OUTPUT   = joinpath(EXPERIMENTS_DIR, "grid_search_results.csv")

function assemble()
    all_rows = String[]
    header = nothing

    for prob in PROBLEMS
        for batch in BATCHES
            summary = joinpath(EXPERIMENTS_DIR, prob, batch, "summary.csv")
            if !isfile(summary)
                println("WARNING: missing $summary")
                continue
            end
            lines = readlines(summary)
            if isempty(lines)
                continue
            end
            if header === nothing
                header = lines[1]
            end
            # Skip header, add data rows
            for line in lines[2:end]
                if !isempty(strip(line))
                    push!(all_rows, line)
                end
            end
            println("Read $(length(lines)-1) rows from $summary")
        end
    end

    if header === nothing
        println("No summary files found!")
        return
    end

    open(OUTPUT, "w") do io
        println(io, header)
        for row in all_rows
            println(io, row)
        end
    end
    println("\nAssembled $(length(all_rows)) rows into $OUTPUT")
end

assemble()
