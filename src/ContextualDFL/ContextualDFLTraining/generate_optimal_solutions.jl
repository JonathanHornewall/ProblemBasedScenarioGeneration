#!/usr/bin/env julia

using ArgParse
using ContextualDFLExperiments
using ContextualDFLTraining

function parse_commandline(args=ARGS)
    settings = ArgParseSettings(
        description="Generate precomputed optimal solutions for one ContextualDFLTraining experiment.",
    )

    @add_arg_table! settings begin
        "--experiment"
            help = "Experiment id, module name, or config path, e.g. resource_allocation/experiment_1"
            required = true
        "--splits"
            help = "Comma-separated split names to generate; defaults to every optimality split"
            default = ""
    end

    return parse_args(args, settings)
end

function requested_splits(parsed_args)
    raw = parsed_args["splits"]
    isempty(strip(raw)) && return nothing
    return Set(Symbol(strip(value)) for value in split(raw, ",") if !isempty(strip(value)))
end

function selected_experiment(parsed_args)
    return ContextualDFLTraining.load_experiment(parsed_args["experiment"])
end

function main()
    parsed_args = parse_commandline()
    experiment = selected_experiment(parsed_args)
    config = ContextualDFLTraining.experiment_base_config(experiment)
    objects = ContextualDFLTraining.experiment_call(experiment, :training_objects, config)
    split_filter = requested_splits(parsed_args)
    splits = ContextualDFLTraining.experiment_call(experiment, :optimality_splits, objects, config)
    evaluation_batches = something(
        ContextualDFLTraining.config_value(config, :optimality_evaluation_batches, 1),
        1,
    )
    generated = Symbol[]

    for (split_name, dataset) in splits
        split_name = Symbol(split_name)
        split_filter !== nothing && !(split_name in split_filter) && continue
        isempty(dataset) && continue
        if split_name == :test && ContextualDFLTraining.uses_generated_test_data(experiment)
            println(
                "Skipping test split for experiment=$(experiment.id); generate_test_data.jl owns generated test-data optimal solutions.",
            )
            continue
        end

        println(
            "Computing optimal results for experiment=$(experiment.id), split=$(split_name), samples=$(length(dataset))",
        )
        results = nothing
        solve_seconds = @elapsed begin
            results = ContextualDFLExperiments.solve_dataset_to_optimality(
                dataset,
                objects.program,
                objects.reference_scenario_decoder,
                objects.solver;
                mu=Float64(ContextualDFLTraining.config_value(config, :optimality_mu, 0.0)),
                rho=Float64(ContextualDFLTraining.config_value(config, :optimality_rho, 0.0)),
                evaluation_batches=evaluation_batches,
            )
        end
        path = ContextualDFLTraining.save_optimal_results!(
            experiment,
            split_name,
            results;
            dataset=dataset,
            metadata=(; solve_seconds=solve_seconds, evaluation_batches=evaluation_batches),
        )
        println("Wrote optimal results to $path")

        path = ContextualDFLTraining.optimal_results_path(experiment, split_name)
        payload_results = ContextualDFLTraining.load_optimal_results(
            experiment,
            split_name;
            dataset=dataset,
        )
        length(payload_results) == length(dataset) ||
            error("saved optimal results at $path have the wrong length")
        println("Finished split=$(split_name) in $(round(solve_seconds; digits=3)) seconds")
        push!(generated, split_name)
    end

    isempty(generated) && println("No optimal-result splits were generated.")
    return generated
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
