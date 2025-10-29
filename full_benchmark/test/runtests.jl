using Test

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const TEST_KEY = "FULL_BENCHMARK_TEST_MODE"
const PREV_TEST_MODE = haskey(ENV, TEST_KEY) ? ENV[TEST_KEY] : nothing
ENV[TEST_KEY] = "1"

const RunExperiment = Module(:RunExperiment)
Base.include(RunExperiment, joinpath(PROJECT_ROOT, "scripts", "run_experiment.jl"))
const run_experiment_main = RunExperiment.main

function run_pipeline(args::Vector{String})
    run_experiment_main(args)
end

try
    @testset "Full Benchmark Smoke" begin
        mktempdir() do output_dir
            args = [
                "--output-dir=$(output_dir)",
                "--training-size=2",
                "--testing-covariates=1",
                "--testing-scenarios=2",
                "--testing-collections=1",
                "--training-scenarios-per-context=1"
            ]
            run_pipeline(args)
            @test isfile(joinpath(output_dir, "artifacts", "results", "benchmark_gaps.csv"))
            @test isfile(joinpath(output_dir, "artifacts", "models", "neural", "neural_model_final.jls"))
        end
    end

    @testset "Flag Behaviour" begin
        mktempdir() do base_output
            args1 = [
                "--output-dir=$(base_output)",
                "--training-size=2",
                "--testing-covariates=1",
                "--testing-scenarios=2",
                "--testing-collections=1",
                "--training-scenarios-per-context=1"
            ]
            run_pipeline(args1)

            covariates_base = read(joinpath(base_output, "artifacts", "training", "training_covariates.csv"), String)
            report_base = read(joinpath(base_output, "artifacts", "models", "baselines", "baseline_training_report.json"), String)

            mktempdir() do method_output
                args2 = [
                    "--input-dir=$(base_output)",
                    "--output-dir=$(method_output)",
                    "--method_training"
                ]
                run_pipeline(args2)

                covariates_method = read(joinpath(method_output, "artifacts", "training", "training_covariates.csv"), String)
                report_method = read(joinpath(method_output, "artifacts", "models", "baselines", "baseline_training_report.json"), String)

                @test covariates_base == covariates_method
                @test report_base != report_method
                @test isfile(joinpath(method_output, "artifacts", "results", "benchmark_gaps.csv"))
            end
        end
    end
finally
    if PREV_TEST_MODE === nothing
        delete!(ENV, TEST_KEY)
    else
        ENV[TEST_KEY] = PREV_TEST_MODE
    end
end
