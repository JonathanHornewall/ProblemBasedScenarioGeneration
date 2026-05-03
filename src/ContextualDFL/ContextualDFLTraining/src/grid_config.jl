const DEFAULT_GRID_VALUES = (;
    learning_rate=[1e-3, 5e-4, 3e-4],
    hidden_size=[64, 128],
    depth=[2, 3],
    batch_size=[8, 16],
    dropout=[0.0],
    seed=[143],
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
    settings = _merge_settings((; epochs=2, overrides...))
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
