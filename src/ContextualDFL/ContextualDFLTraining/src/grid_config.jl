const DEFAULT_RUN_SETTINGS = (;
    epochs=100,
    n_samples=2000,
    validation_fraction=0.15,
    test_fraction=0.15,
    sigma=5.0,
    demand_power=1.0,
    context_terms=3,
    mu=1e-1,
    mu_start=1e-1,
    mu_end=1e-3,
    mu_schedule=:geometric,
    nr_scenarios=1,
    rho=0.0,
    tolerance_relative=0.10,
    tolerance_absolute_floor=1.0,
    optimality_evaluation=false,
    optimality_test_sample_count=0,
    optimality_train_sample_count=0,
    optimality_validation_sample_count=0,
    optimality_mu=0.0,
    solver=:highs,
    loss=:dfl_scen,
)

const DEFAULT_GRID_VALUES = (;
    learning_rate=[1e-4, 3e-4, 1e-3],
    hidden_size=[64, 128, 256],
    depth=[2, 3],
    batch_size=[32, 64],
    dropout=[0.0, 0.1],
    seed=[1, 2, 3],
)

function _merge_settings(overrides::NamedTuple)
    return merge(DEFAULT_RUN_SETTINGS, overrides)
end

function _run_id(index, cfg)
    return "run_" * lpad(string(index), 4, "0") * "_seed" * string(cfg.seed)
end

function default_grid(; overrides...)
    settings = _merge_settings(NamedTuple(overrides))
    configs = NamedTuple[]
    index = 0

    for learning_rate in DEFAULT_GRID_VALUES.learning_rate,
        hidden_size in DEFAULT_GRID_VALUES.hidden_size,
        depth in DEFAULT_GRID_VALUES.depth,
        batch_size in DEFAULT_GRID_VALUES.batch_size,
        dropout in DEFAULT_GRID_VALUES.dropout,
        seed in DEFAULT_GRID_VALUES.seed

        index += 1
        cfg = merge(
            settings,
            (;
                learning_rate=Float64(learning_rate),
                hidden_size=Int(hidden_size),
                depth=Int(depth),
                batch_size=Int(batch_size),
                dropout=Float64(dropout),
                seed=Int(seed),
            ),
        )
        push!(configs, merge(cfg, (; run_id=_run_id(index, cfg))))
    end

    return configs
end

function smoke_grid(; overrides...)
    settings = _merge_settings((; epochs=2, n_samples=16, overrides...))
    cfg = merge(
        settings,
        (;
            learning_rate=1e-3,
            hidden_size=16,
            depth=1,
            batch_size=4,
            dropout=0.0,
            seed=1,
        ),
    )
    return [merge(cfg, (; run_id="smoke_0001_seed1"))]
end
