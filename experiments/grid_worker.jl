#!/usr/bin/env julia
"""
Grid search worker — trains multiple configs for a single problem in one Julia process.

Usage:
    julia --project=../src/ProblemBasedScenarioGeneration grid_worker.jl \
        --problem newsvendor \
        --output-dir newsvendor/batch_a \
        --configs "64:2:relu,64:2:tanh,64:3:relu,64:3:tanh,64:4:relu,64:4:tanh,128:2:relu,128:2:tanh"

Each config is hidden_dim:n_layers:activation separated by commas.
"""

using Pkg
# Activate the package
Pkg.activate(joinpath(@__DIR__, "..", "src", "ProblemBasedScenarioGeneration"))

using ProblemBasedScenarioGeneration
using Flux
using Random
using Statistics
using Dates

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

function parse_worker_args(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        if args[i] == "--problem"
            d["problem"] = args[i+1]; i += 2
        elseif args[i] == "--output-dir"
            d["output_dir"] = args[i+1]; i += 2
        elseif args[i] == "--configs"
            d["configs"] = args[i+1]; i += 2
        elseif args[i] == "--epochs"
            d["epochs"] = args[i+1]; i += 2
        elseif args[i] == "--n-train"
            d["n_train"] = args[i+1]; i += 2
        elseif args[i] == "--n-test"
            d["n_test"] = args[i+1]; i += 2
        else
            error("Unknown argument: $(args[i])")
        end
    end
    return d
end

function parse_configs(s::String)
    configs = []
    for part in split(s, ",")
        tokens = split(part, ":")
        hidden_dim = parse(Int, tokens[1])
        n_layers   = parse(Int, tokens[2])
        activation = String(tokens[3])
        push!(configs, (hidden_dim=hidden_dim, n_layers=n_layers, activation=activation))
    end
    return configs
end

# ---------------------------------------------------------------------------
# Fixed training parameters
# ---------------------------------------------------------------------------

const DEFAULT_EPOCHS  = 50
const LR              = 1e-3
const BATCHSIZE       = 1
const DEFAULT_N_TRAIN = 200
const DEFAULT_N_TEST  = 50
const MU_SURR         = 1.0
const MU_PRIM         = 1.0  # 0.0 causes gradient explosion; 1.0 is numerically stable
const SEED            = 42
const SIGMA           = 5.0

# ---------------------------------------------------------------------------
# Main worker logic
# ---------------------------------------------------------------------------

function run_worker()
    wa = parse_worker_args(ARGS)
    problem_name = wa["problem"]
    base_dir     = joinpath(@__DIR__, wa["output_dir"])
    configs      = parse_configs(wa["configs"])

    # Allow per-run overrides of epochs/n_train/n_test
    EPOCHS  = haskey(wa, "epochs")  ? parse(Int, wa["epochs"])  : DEFAULT_EPOCHS
    N_TRAIN = haskey(wa, "n_train") ? parse(Int, wa["n_train"]) : DEFAULT_N_TRAIN
    N_TEST  = haskey(wa, "n_test")  ? parse(Int, wa["n_test"])  : DEFAULT_N_TEST

    # Resolve problem
    prob = ProblemBasedScenarioGeneration.resolve_problem(problem_name)
    println("=" ^ 60)
    println("Grid worker: $problem_name ($(length(configs)) configs)")
    println("Epochs: $EPOCHS | N_TRAIN: $N_TRAIN | N_TEST: $N_TEST")
    println("Output: $base_dir")
    println("=" ^ 60)

    # Generate dataset once (same seed, same data for all configs in this batch)
    Random.seed!(SEED)
    if prob isa ProblemBasedScenarioGeneration.UnreliableNewsvendorProblem
        train_data = generate_dataset(prob, N_TRAIN)
        test_data  = generate_dataset(prob, N_TEST)
    else
        train_data = generate_dataset(prob, N_TRAIN; sigma=SIGMA)
        test_data  = generate_dataset(prob, N_TEST; sigma=SIGMA)
    end

    # Summary rows
    summary_rows = []

    for (i, cfg) in enumerate(configs)
        run_name = "h$(cfg.hidden_dim)_l$(cfg.n_layers)_$(cfg.activation)"
        run_dir  = joinpath(base_dir, run_name)
        mkpath(run_dir)

        println("\n--- Config $i/$(length(configs)): $run_name ---")
        println("  hidden_dim=$(cfg.hidden_dim), n_layers=$(cfg.n_layers), activation=$(cfg.activation)")

        try
            # Set seed for reproducibility per config
            Random.seed!(SEED)

            # Build model
            act_func = ProblemBasedScenarioGeneration._activation_func(cfg.activation)
            generator = build_generator(
                prob;
                hidden_dim = cfg.hidden_dim,
                n_layers   = cfg.n_layers,
                activation = act_func,
            )
            println("  Model: $generator")

            # Train
            t0 = time()
            opt = Adam(LR)
            loss_history = train!(
                generator, prob, train_data;
                mu_surr   = MU_SURR,
                mu_prim   = MU_PRIM,
                opt       = opt,
                epochs    = EPOCHS,
                batchsize = BATCHSIZE,
                verbose   = true,
            )
            elapsed = time() - t0
            println("  Training time: $(round(elapsed; digits=1))s")

            final_loss = loss_history[end]
            min_loss   = minimum(loss_history)
            println("  Final loss: $final_loss")
            println("  Min loss:   $min_loss")

            # Save loss.csv
            loss_csv = joinpath(run_dir, "loss.csv")
            open(loss_csv, "w") do io
                println(io, "epoch,loss")
                for (ep, l) in enumerate(loss_history)
                    println(io, "$ep,$l")
                end
            end

            # Save checkpoint
            config_dict = Dict{String,Any}(
                "mu_surr"    => MU_SURR,
                "mu_prim"    => MU_PRIM,
                "lr"         => LR,
                "optimizer"  => "adam",
                "batchsize"  => BATCHSIZE,
                "epochs"     => EPOCHS,
                "n_train"    => N_TRAIN,
                "sigma"      => SIGMA,
                "seed"       => SEED,
                "hidden_dim" => cfg.hidden_dim,
                "n_layers"   => cfg.n_layers,
                "activation" => cfg.activation,
                "total_epochs" => EPOCHS,
                "continuation" => false,
            )
            model_path = joinpath(run_dir, "model.jls")
            save_checkpoint(model_path, generator, prob, config_dict, loss_history)
            println("  Saved: $model_path")

            # Quick test evaluation
            mean_regret = NaN
            try
                regrets = Float64[]
                for (x, actual_sc) in test_data
                    raw = generator(x)
                    pred = ProblemBasedScenarioGeneration._params_to_scenarios(prob, raw)
                    r = decision_regret(prob, MU_SURR, MU_PRIM, pred, actual_sc)
                    push!(regrets, r)
                end
                mean_regret = mean(regrets)
                println("  Mean test regret: $mean_regret")
            catch e
                println("  Test evaluation failed: $e")
            end

            push!(summary_rows, (
                problem      = problem_name,
                hidden_dim   = cfg.hidden_dim,
                n_layers     = cfg.n_layers,
                activation   = cfg.activation,
                final_loss   = final_loss,
                min_loss     = min_loss,
                mean_regret  = mean_regret,
                epochs       = EPOCHS,
                elapsed_s    = round(elapsed; digits=1),
            ))

        catch e
            println("  ERROR: Config $run_name failed!")
            println("  ", e)
            push!(summary_rows, (
                problem      = problem_name,
                hidden_dim   = cfg.hidden_dim,
                n_layers     = cfg.n_layers,
                activation   = cfg.activation,
                final_loss   = NaN,
                min_loss     = NaN,
                mean_regret  = NaN,
                epochs       = 0,
                elapsed_s    = 0.0,
            ))
        end
    end

    # Write summary.csv
    summary_path = joinpath(base_dir, "summary.csv")
    open(summary_path, "w") do io
        println(io, "problem,hidden_dim,n_layers,activation,final_loss,min_loss,mean_regret,epochs,elapsed_s")
        for r in summary_rows
            println(io, "$(r.problem),$(r.hidden_dim),$(r.n_layers),$(r.activation),$(r.final_loss),$(r.min_loss),$(r.mean_regret),$(r.epochs),$(r.elapsed_s)")
        end
    end
    println("\n\nSummary saved: $summary_path")

    # Verify outputs
    println("\n--- Verification ---")
    ok = true
    for cfg in configs
        run_name = "h$(cfg.hidden_dim)_l$(cfg.n_layers)_$(cfg.activation)"
        run_dir  = joinpath(base_dir, run_name)
        for f in ["model.jls", "loss.csv"]
            path = joinpath(run_dir, f)
            if !isfile(path)
                println("  MISSING: $path")
                ok = false
            end
        end
    end
    if ok
        println("  All $(length(configs)) configs produced model.jls and loss.csv ✓")
    end

    # Verify one checkpoint loads (find the first one that exists)
    verified = false
    for cfg in configs
        run_name = "h$(cfg.hidden_dim)_l$(cfg.n_layers)_$(cfg.activation)"
        test_ckpt = joinpath(base_dir, run_name, "model.jls")
        if isfile(test_ckpt)
            println("  Loading checkpoint: $test_ckpt")
            ckpt = load_checkpoint(test_ckpt)
            restored = restore_model(ckpt)
            println("  Checkpoint loads OK: $restored ✓")
            verified = true
            break
        end
    end
    if !verified
        println("  WARNING: No checkpoints to verify (all configs may have failed)")
    end

    println("\nWorker complete!")
end

run_worker()
