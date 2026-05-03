import ContextualDFLTraining

const EXPERIMENT_ID = "resource_allocation/experiment_1"
const EXPERIMENT_NAME = "resource_allocation_experiment_1"

experiment_id() = EXPERIMENT_ID
experiment_name() = EXPERIMENT_NAME
experiment_module_name() = :ResourceAllocationExperiment1
artifact_dir() = joinpath(@__DIR__, "artifacts")

function experiment_overrides(; overrides...)
    return merge(
        (;
            experiment_id=EXPERIMENT_ID,
            experiment_name=EXPERIMENT_NAME,
            problem=:resource_allocation,
            mlflow_dataset_name=EXPERIMENT_NAME,
        ),
        NamedTuple(overrides),
    )
end

function base_config(; overrides...)
    return merge(
        ContextualDFLTraining.DEFAULT_RUN_SETTINGS,
        experiment_overrides(; overrides...),
    )
end

function grid_configs(; overrides...)
    return ContextualDFLTraining.default_grid(; experiment_overrides(; overrides...)...)
end

function smoke_configs(; overrides...)
    return ContextualDFLTraining.smoke_grid(; experiment_overrides(; overrides...)...)
end

function profile_config(; overrides...)
    return merge(
        base_config(;
            epochs=100,
            warmup_epochs=2,
            n_samples=2000,
            learning_rate=1e-3,
            hidden_size=128,
            depth=2,
            batch_size=64,
            dropout=0.0,
            seed=3,
            run_id="profile_standard_seed3",
            base_run_id="profile_standard_seed3",
            candidate_name="profile_standard_seed3",
            method_variant="profiling",
            overrides...,
        ),
        (;
            mlflow_dataset_context="profiling",
            mlflow_source_name="ContextualDFLTraining/profile_training.jl",
            mlflow_params=(;
                profile_target="ContextualDFL.train!",
                profile_loss="ContextualDFL.DflScenLoss",
                profile_progress_logged_by="remote_worker",
            ),
        ),
    )
end

function training_objects(config)
    return ContextualDFLTraining.resource_allocation_training_objects(config)
end

function optimality_splits(objects, config)
    return ContextualDFLTraining.optimality_evaluation_datasets(objects, config)
end

function optimal_results_path(split_name::Symbol)
    return joinpath(artifact_dir(), "optimal_solutions", string(split_name) * ".jls")
end
