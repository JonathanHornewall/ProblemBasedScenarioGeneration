"""
Command-line interface for ProblemBasedScenarioGeneration.

Subcommands: train, continue, test, evaluate, info.

Usage:
    julia bin/pbsg.jl train -p newsvendor --epochs 10 -v -o model.jls
    julia bin/pbsg.jl info -c model.jls
    julia bin/pbsg.jl test -c model.jls --n-test 50
    julia bin/pbsg.jl evaluate -c model.jls --n-test 100 -d ./eval_results
    julia bin/pbsg.jl continue -c model.jls --epochs 5 -o model_v2.jls
"""

using ArgParse
using Dates
using Random
using Statistics
using Flux

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

function resolve_optimizer(name::String, lr::Float64)
    if name == "adam"
        return Adam(lr)
    elseif name == "sgd"
        return Descent(lr)
    else
        error("Unknown optimizer: $name. Choose from: adam, sgd")
    end
end

function resolve_activation(name::String)
    return _activation_func(name)
end

function auto_checkpoint_name(problem_name::String)
    ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    return "pbsg_$(problem_name)_$(ts).jls"
end

function _generate_dataset_for_problem(prob::ProblemInstance, n::Int; sigma::Float64=5.0)
    if prob isa UnreliableNewsvendorProblem
        return generate_dataset(prob, n)
    else
        return generate_dataset(prob, n; sigma=sigma)
    end
end

function _parse_mu_schedule(s::String)
    return parse.(Float64, split(s, ","))
end

# -----------------------------------------------------------------------
# ArgParse settings
# -----------------------------------------------------------------------

function build_arg_parser()
    s = ArgParseSettings(
        prog = "pbsg",
        commands_are_required = false,
        description = "ProblemBasedScenarioGeneration CLI\n\n" *
            "Train, evaluate, and manage neural scenario generators for\n" *
            "two-stage stochastic linear programming.\n\n" *
            "Available problems:\n" *
            "  resource_allocation   20 clients, 30 resources (H_ONLY noise)\n" *
            "  shipment_planning     12 warehouses, 4 locations (H_ONLY noise)\n" *
            "  newsvendor            Unreliable newsvendor with demand + reliability (WH noise)",
        epilog = "examples:\n" *
            "  julia bin/pbsg.jl train -p newsvendor --epochs 30 -v -o model.jls\n" *
            "  julia bin/pbsg.jl train -p resource_allocation --continuation -v\n" *
            "  julia bin/pbsg.jl info -c model.jls\n" *
            "  julia bin/pbsg.jl test -c model.jls --n-test 100\n" *
            "  julia bin/pbsg.jl evaluate -c model.jls -d eval_out --csv -v\n" *
            "  julia bin/pbsg.jl continue -c model.jls --epochs 10 -v",
    )

    # ---------------------------------------------------------------
    # Subcommands
    # ---------------------------------------------------------------
    @add_arg_table! s begin
        "train"
            action = :command
            help = "Train a new scenario generator model"
        "continue"
            action = :command
            help = "Resume training from a checkpoint"
        "test"
            action = :command
            help = "Evaluate a trained model on test data"
        "evaluate"
            action = :command
            help = "Full evaluation with metrics, plots, and CSV export"
        "info"
            action = :command
            help = "Show checkpoint metadata and training history"
    end

    # ---------------------------------------------------------------
    # train
    # ---------------------------------------------------------------
    s_train = s["train"]
    s_train.description = "Train a scenario generator from scratch.\n\n" *
        "The model is a feed-forward neural network (Flux.Chain) that maps\n" *
        "context vectors to scenario parameters. Training minimises decision\n" *
        "regret: the cost gap between decisions made with predicted vs. actual\n" *
        "scenarios."
    s_train.epilog = "examples:\n" *
        "  # Minimal training run\n" *
        "  julia bin/pbsg.jl train -p newsvendor --epochs 5 --n-train 50 -v\n\n" *
        "  # Custom architecture and optimizer\n" *
        "  julia bin/pbsg.jl train -p resource_allocation \\\n" *
        "      --hidden-dim 256 --n-layers 4 --activation tanh \\\n" *
        "      --lr 5e-4 --epochs 50 -v -o ra_model.jls\n\n" *
        "  # Continuation schedule (anneal barrier parameter)\n" *
        "  julia bin/pbsg.jl train -p shipment_planning --continuation \\\n" *
        "      --mu-schedule \"1.0,0.5,0.1,0.01\" --epochs-per-stage 5 -v\n\n" *
        "  # Save checkpoints every 10 epochs and log losses\n" *
        "  julia bin/pbsg.jl train -p newsvendor --epochs 100 \\\n" *
        "      --save-interval 10 --loss-log losses.csv -o model.jls\n\n" *
        "  # Reproducible run with fixed seed\n" *
        "  julia bin/pbsg.jl train -p newsvendor --seed 42 --epochs 20 -v"

    add_arg_group!(s_train, "Problem", "Problem")
    @add_arg_table! s_train begin
        "--problem", "-p"
            help = "Problem type: resource_allocation, shipment_planning, or newsvendor"
            required = true
            group = "Problem"
        "--nr-scenarios"
            help = "Number of scenarios the model outputs per prediction"
            arg_type = Int
            default = 1
            group = "Problem"
    end

    add_arg_group!(s_train, "Data generation", "Data generation")
    @add_arg_table! s_train begin
        "--n-train"
            help = "Number of training (context, scenario) pairs to generate"
            arg_type = Int
            default = 200
            group = "Data generation"
        "--n-test"
            help = "Number of test samples for post-training evaluation (0 to skip)"
            arg_type = Int
            default = 50
            group = "Data generation"
        "--sigma"
            help = "Noise standard deviation for data generation (ignored for newsvendor)"
            arg_type = Float64
            default = 5.0
            group = "Data generation"
    end

    add_arg_group!(s_train, "Model architecture", "Model architecture")
    @add_arg_table! s_train begin
        "--hidden-dim"
            help = "Width of hidden layers"
            arg_type = Int
            default = 128
            group = "Model architecture"
        "--n-layers"
            help = "Number of hidden layers (output layer is added automatically)"
            arg_type = Int
            default = 3
            group = "Model architecture"
        "--activation"
            help = "Hidden-layer activation function: relu, tanh, sigmoid, or softplus"
            default = "relu"
            group = "Model architecture"
    end

    add_arg_group!(s_train, "Optimization", "Optimization")
    @add_arg_table! s_train begin
        "--epochs"
            help = "Number of training epochs (full passes over the dataset)"
            arg_type = Int
            default = 30
            group = "Optimization"
        "--batchsize"
            help = "Mini-batch size (1 = pure SGD, higher = mini-batch SGD)"
            arg_type = Int
            default = 1
            group = "Optimization"
        "--lr"
            help = "Learning rate for the optimizer"
            arg_type = Float64
            default = 1e-3
            group = "Optimization"
        "--optimizer"
            help = "Optimizer algorithm: adam (Adam) or sgd (gradient descent)"
            default = "adam"
            group = "Optimization"
    end

    add_arg_group!(s_train, "Barrier parameters", "Barrier parameters")
    @add_arg_table! s_train begin
        "--mu-surr"
            help = "Log-barrier parameter for the surrogate LP solve (smoothing level)"
            arg_type = Float64
            default = 1.0
            group = "Barrier parameters"
        "--mu-prim"
            help = "Log-barrier parameter for cost evaluation (0 = exact LP)"
            arg_type = Float64
            default = 0.0
            group = "Barrier parameters"
    end

    add_arg_group!(s_train, "Continuation schedule (--continuation)", "Continuation schedule (--continuation)")
    @add_arg_table! s_train begin
        "--continuation"
            help = "Enable 3-phase mu-continuation: warmup -> anneal -> finetune"
            action = :store_true
            group = "Continuation schedule (--continuation)"
        "--mu-schedule"
            help = "Comma-separated list of decreasing mu values for the annealing phase"
            default = "1.0,0.8,0.6,0.4,0.2,0.1,0.08,0.06,0.04,0.02,0.01"
            group = "Continuation schedule (--continuation)"
        "--epochs-per-stage"
            help = "Training epochs at each mu value during annealing"
            arg_type = Int
            default = 10
            group = "Continuation schedule (--continuation)"
        "--warmup-epochs"
            help = "Epochs for the warm-up phase (train at largest mu)"
            arg_type = Int
            default = 20
            group = "Continuation schedule (--continuation)"
        "--finetune-epochs"
            help = "Epochs for the fine-tuning phase (train at mu=0, exact LP)"
            arg_type = Int
            default = 10
            group = "Continuation schedule (--continuation)"
    end

    add_arg_group!(s_train, "Output and logging", "Output and logging")
    @add_arg_table! s_train begin
        "--output", "-o"
            help = "Path to save the final checkpoint (.jls). Auto-generated if omitted"
            default = ""
            group = "Output and logging"
        "--save-interval"
            help = "Save an intermediate checkpoint every N epochs (0 = final only)"
            arg_type = Int
            default = 0
            group = "Output and logging"
        "--loss-log"
            help = "Write per-epoch loss to this CSV file (columns: epoch, loss)"
            default = ""
            group = "Output and logging"
        "--verbose", "-v"
            help = "Print per-epoch loss and training phase transitions"
            action = :store_true
            group = "Output and logging"
        "--seed"
            help = "Random seed for reproducibility (-1 = no fixed seed)"
            arg_type = Int
            default = -1
            group = "Output and logging"
    end

    # ---------------------------------------------------------------
    # continue
    # ---------------------------------------------------------------
    s_cont = s["continue"]
    s_cont.description = "Resume training from a previously saved checkpoint.\n\n" *
        "Loads the model architecture and weights, re-generates the training\n" *
        "dataset (same size and sigma as the original run), and trains for\n" *
        "additional epochs. All unspecified parameters default to the values\n" *
        "stored in the checkpoint."
    s_cont.epilog = "examples:\n" *
        "  # Continue for 20 more epochs with same settings\n" *
        "  julia bin/pbsg.jl continue -c model.jls --epochs 20 -v\n\n" *
        "  # Continue with a lower learning rate\n" *
        "  julia bin/pbsg.jl continue -c model.jls --epochs 10 --lr 1e-4\n\n" *
        "  # Save to a new file\n" *
        "  julia bin/pbsg.jl continue -c model.jls --epochs 10 -o model_v2.jls"

    @add_arg_table! s_cont begin
        "--checkpoint", "-c"
            help = "Path to the checkpoint (.jls) to resume from"
            required = true
        "--epochs"
            help = "Number of additional epochs to train"
            arg_type = Int
            default = 30
        "--lr"
            help = "Learning rate override (-1 = use checkpoint value)"
            arg_type = Float64
            default = -1.0
        "--output", "-o"
            help = "Path for the new checkpoint (default: <original>_continued.jls)"
            default = ""
        "--mu-surr"
            help = "Surrogate barrier parameter override (-1 = use checkpoint value)"
            arg_type = Float64
            default = -1.0
        "--mu-prim"
            help = "Evaluation barrier parameter override (-1 = use checkpoint value)"
            arg_type = Float64
            default = -1.0
        "--batchsize"
            help = "Mini-batch size override (-1 = use checkpoint value)"
            arg_type = Int
            default = -1
        "--verbose", "-v"
            help = "Print per-epoch loss during training"
            action = :store_true
        "--seed"
            help = "Random seed (-1 = no fixed seed)"
            arg_type = Int
            default = -1
    end

    # ---------------------------------------------------------------
    # test
    # ---------------------------------------------------------------
    s_test = s["test"]
    s_test.description = "Evaluate a trained model on freshly generated test data.\n\n" *
        "Loads the model from a checkpoint, generates test (context, scenario)\n" *
        "pairs, and computes decision regret and relative decision regret for\n" *
        "each sample. Prints summary statistics (mean, std, min, max)."
    s_test.epilog = "examples:\n" *
        "  # Evaluate with 100 test samples\n" *
        "  julia bin/pbsg.jl test -c model.jls --n-test 100\n\n" *
        "  # Save per-sample results to CSV\n" *
        "  julia bin/pbsg.jl test -c model.jls --n-test 200 -o results.csv\n\n" *
        "  # Evaluate with exact LP (mu=0)\n" *
        "  julia bin/pbsg.jl test -c model.jls --mu-prim 0.0\n\n" *
        "  # Reproducible evaluation\n" *
        "  julia bin/pbsg.jl test -c model.jls --seed 123"

    @add_arg_table! s_test begin
        "--checkpoint", "-c"
            help = "Path to the trained model checkpoint (.jls)"
            required = true
        "--problem", "-p"
            help = "Problem type override (default: inferred from checkpoint metadata)"
            default = ""
        "--n-test"
            help = "Number of test (context, scenario) pairs to generate"
            arg_type = Int
            default = 100
        "--mu-prim"
            help = "Barrier parameter for cost evaluation (0 = exact LP solution)"
            arg_type = Float64
            default = 0.0
        "--seed"
            help = "Random seed for test data generation (-1 = no fixed seed)"
            arg_type = Int
            default = -1
        "--output", "-o"
            help = "Write per-sample results to this CSV file"
            default = ""
    end

    # ---------------------------------------------------------------
    # evaluate
    # ---------------------------------------------------------------
    s_eval = s["evaluate"]
    s_eval.description = "Full evaluation of a trained model with metrics, plots, and CSV.\n\n" *
        "Generates detailed statistics (95% CI, threshold fractions, median)\n" *
        "and publication-ready plots (loss curve, regret histogram, violin,\n" *
        "CDF, scenario scatter). Requires Plots.jl (loaded on first use)."
    s_eval.epilog = "examples:\n" *
        "  # Full evaluation with plots\n" *
        "  julia bin/pbsg.jl evaluate -c model.jls --n-test 100 -d eval_out\n\n" *
        "  # Metrics only (no plots, faster)\n" *
        "  julia bin/pbsg.jl evaluate -c model.jls --no-plots --csv\n\n" *
        "  # PDF plots\n" *
        "  julia bin/pbsg.jl evaluate -c model.jls --format pdf -d plots/"

    @add_arg_table! s_eval begin
        "--checkpoint", "-c"
            help = "Path to the trained model checkpoint (.jls)"
            required = true
        "--n-test"
            help = "Number of test samples to generate"
            arg_type = Int
            default = 100
        "--mu-prim"
            help = "Barrier parameter for cost evaluation (0 = exact LP)"
            arg_type = Float64
            default = 0.0
        "--seed"
            help = "Random seed for test data generation (-1 = no fixed seed)"
            arg_type = Int
            default = -1
        "--output-dir", "-d"
            help = "Directory to save plots and CSV output"
            default = "./eval_results"
        "--format"
            help = "Plot file format: png, pdf, or svg"
            default = "png"
        "--no-plots"
            help = "Skip plot generation (metrics only)"
            action = :store_true
        "--csv"
            help = "Save per-sample metrics to CSV"
            action = :store_true
        "--verbose", "-v"
            help = "Print detailed progress"
            action = :store_true
    end

    # ---------------------------------------------------------------
    # info
    # ---------------------------------------------------------------
    s_info = s["info"]
    s_info.description = "Display metadata stored in a checkpoint file.\n\n" *
        "Shows the model architecture, training hyperparameters, problem type,\n" *
        "loss history summary, and timestamp."
    s_info.epilog = "examples:\n" *
        "  julia bin/pbsg.jl info -c model.jls"

    @add_arg_table! s_info begin
        "--checkpoint", "-c"
            help = "Path to the checkpoint file (.jls) to inspect"
            required = true
    end

    return s
end

# -----------------------------------------------------------------------
# Subcommand: train
# -----------------------------------------------------------------------

function cmd_train(args::Dict)
    # Seed
    seed = args["seed"]
    seed >= 0 && Random.seed!(seed)

    # Problem
    problem_name = args["problem"]
    prob = resolve_problem(problem_name)
    println("Problem: $problem_name ($(noise_pattern(prob)))")

    # Dataset
    n_train = args["n-train"]
    sigma   = args["sigma"]
    println("Generating $n_train training samples...")
    train_data = _generate_dataset_for_problem(prob, n_train; sigma=sigma)

    # Model
    activation = resolve_activation(args["activation"])
    nr_sc = args["nr-scenarios"]
    generator = build_generator(
        prob;
        nr_of_scenarios = nr_sc,
        hidden_dim      = args["hidden-dim"],
        n_layers        = args["n-layers"],
        activation      = activation,
    )
    n_params = _count_params(Dict{String,Any}(
        "input_dim" => _context_dim(prob),
        "output_dim" => _scenario_param_dim(prob) * nr_sc,
        "hidden_dim" => args["hidden-dim"],
        "n_layers" => args["n-layers"],
    ))
    println("Model: $(generator)")
    println("Parameters: $n_params")

    # Optimizer
    lr  = args["lr"]
    opt = resolve_optimizer(args["optimizer"], lr)
    verbose = args["verbose"]

    # Build training config for checkpoint
    config = Dict{String,Any}(
        "mu_surr"          => args["mu-surr"],
        "mu_prim"          => args["mu-prim"],
        "lr"               => lr,
        "optimizer"        => args["optimizer"],
        "batchsize"        => args["batchsize"],
        "epochs"           => args["epochs"],
        "n_train"          => n_train,
        "sigma"            => sigma,
        "seed"             => seed,
        "continuation"     => args["continuation"],
        "mu_schedule"      => args["mu-schedule"],
        "epochs_per_stage" => args["epochs-per-stage"],
        "warmup_epochs"    => args["warmup-epochs"],
        "finetune_epochs"  => args["finetune-epochs"],
        "hidden_dim"       => args["hidden-dim"],
        "n_layers"         => args["n-layers"],
        "activation"       => args["activation"],
        "nr_scenarios"     => nr_sc,
    )

    # Train
    println("\nStarting training...")
    all_losses = Float64[]
    save_interval = args["save-interval"]

    if args["continuation"]
        mu_schedule = _parse_mu_schedule(args["mu-schedule"])
        all_losses = continuation_train!(
            generator, prob, train_data;
            mu_schedule        = mu_schedule,
            epochs_per_stage   = args["epochs-per-stage"],
            first_stage_epochs = args["warmup-epochs"],
            finetune_epochs    = args["finetune-epochs"],
            opt                = opt,
            verbose            = verbose,
        )
    elseif save_interval > 0
        # Chunked training with periodic saves
        remaining = args["epochs"]
        opt_state = Flux.setup(opt, generator)
        stage = 1
        while remaining > 0
            chunk = min(save_interval, remaining)
            losses = train!(generator, prob, train_data;
                mu_surr   = args["mu-surr"],
                mu_prim   = args["mu-prim"],
                opt       = opt,
                epochs    = chunk,
                batchsize = args["batchsize"],
                verbose   = verbose,
                opt_state = opt_state,
            )
            append!(all_losses, losses)
            remaining -= chunk

            # Save intermediate checkpoint
            output_path = args["output"]
            if isempty(output_path)
                output_path = auto_checkpoint_name(problem_name)
            end
            base, ext = splitext(output_path)
            interim_path = "$(base)_stage$(stage)$(ext)"
            config["total_epochs"] = length(all_losses)
            save_checkpoint(interim_path, generator, prob, config, all_losses; nr_of_scenarios=nr_sc)
            verbose && println("Saved checkpoint: $interim_path")
            stage += 1
        end
    else
        all_losses = train!(generator, prob, train_data;
            mu_surr   = args["mu-surr"],
            mu_prim   = args["mu-prim"],
            opt       = opt,
            epochs    = args["epochs"],
            batchsize = args["batchsize"],
            verbose   = verbose,
        )
    end

    config["total_epochs"] = length(all_losses)

    # Save final checkpoint
    output_path = args["output"]
    if isempty(output_path)
        output_path = auto_checkpoint_name(problem_name)
    end
    save_checkpoint(output_path, generator, prob, config, all_losses; nr_of_scenarios=nr_sc)
    println("\nCheckpoint saved: $output_path")

    # Write loss log if requested
    loss_log = args["loss-log"]
    if !isempty(loss_log)
        open(loss_log, "w") do io
            println(io, "epoch,loss")
            for (i, l) in enumerate(all_losses)
                println(io, "$i,$l")
            end
        end
        println("Loss log saved: $loss_log")
    end

    # Quick evaluation on test set
    n_test = args["n-test"]
    if n_test > 0
        println("\nEvaluating on $n_test test samples...")
        test_data = _generate_dataset_for_problem(prob, n_test; sigma=sigma)
        mu_prim_eval = args["mu-prim"]
        mu_surr_eval = args["mu-surr"]

        regrets = Float64[]
        for (x, actual_sc) in test_data
            raw = generator(x)
            pred = _params_to_scenarios(prob, raw)
            r = decision_regret(prob, mu_surr_eval, mu_prim_eval, pred, actual_sc)
            push!(regrets, r)
        end

        println("  Mean regret:   $(mean(regrets))")
        println("  Std regret:    $(std(regrets))")
    end

    println("\nTraining complete!")
    return all_losses
end

# -----------------------------------------------------------------------
# Subcommand: continue
# -----------------------------------------------------------------------

function cmd_continue(args::Dict)
    # Load checkpoint
    ckpt_path = args["checkpoint"]
    println("Loading checkpoint: $ckpt_path")
    ckpt = load_checkpoint(ckpt_path)
    tc = get(ckpt, "training_config", Dict())

    # Seed
    seed = args["seed"]
    seed >= 0 && Random.seed!(seed)

    # Restore model
    generator = restore_model(ckpt)
    println("Model restored: $(generator)")

    # Resolve problem
    prob = resolve_problem(ckpt["problem_type"])
    println("Problem: $(ckpt["problem_type"])")

    # Override parameters from CLI or fall back to checkpoint
    lr       = args["lr"] > 0       ? args["lr"]       : get(tc, "lr", 1e-3)
    mu_surr  = args["mu-surr"] >= 0 ? args["mu-surr"]  : get(tc, "mu_surr", 1.0)
    mu_prim  = args["mu-prim"] >= 0 ? args["mu-prim"]  : get(tc, "mu_prim", 0.0)
    batchsize = args["batchsize"] > 0 ? args["batchsize"] : get(tc, "batchsize", 1)
    epochs   = args["epochs"]
    verbose  = args["verbose"]

    optimizer_name = get(tc, "optimizer", "adam")
    opt = resolve_optimizer(optimizer_name, lr)

    # Re-generate dataset
    n_train = get(tc, "n_train", 200)
    sigma   = get(tc, "sigma", 5.0)
    println("Generating $n_train training samples...")
    train_data = _generate_dataset_for_problem(prob, n_train; sigma=sigma)

    # Train
    prev_epochs = get(ckpt, "total_epochs", length(get(ckpt, "loss_history", Float64[])))
    println("\nContinuing training for $epochs epochs (from epoch $prev_epochs)...")

    losses = train!(generator, prob, train_data;
        mu_surr   = mu_surr,
        mu_prim   = mu_prim,
        opt       = opt,
        epochs    = epochs,
        batchsize = batchsize,
        verbose   = verbose,
    )

    # Update checkpoint config
    new_config = copy(tc)
    new_config["lr"]           = lr
    new_config["mu_surr"]      = mu_surr
    new_config["mu_prim"]      = mu_prim
    new_config["batchsize"]    = batchsize
    new_config["total_epochs"] = prev_epochs + epochs

    all_losses = vcat(get(ckpt, "loss_history", Float64[]), losses)

    # Save new checkpoint
    output_path = args["output"]
    if isempty(output_path)
        base, ext = splitext(ckpt_path)
        output_path = "$(base)_continued$(ext)"
    end
    nr_sc = get(ckpt, "nr_of_scenarios", 1)
    save_checkpoint(output_path, generator, prob, new_config, all_losses; nr_of_scenarios=nr_sc)
    println("\nCheckpoint saved: $output_path")
    println("Total epochs: $(prev_epochs + epochs)")
    println("Final loss: $(losses[end])")

    return losses
end

# -----------------------------------------------------------------------
# Subcommand: test
# -----------------------------------------------------------------------

function cmd_test(args::Dict)
    # Load checkpoint
    ckpt_path = args["checkpoint"]
    println("Loading checkpoint: $ckpt_path")
    ckpt = load_checkpoint(ckpt_path)

    # Seed
    seed = args["seed"]
    seed >= 0 && Random.seed!(seed)

    # Restore model
    generator = restore_model(ckpt)

    # Resolve problem
    problem_name = isempty(args["problem"]) ? ckpt["problem_type"] : args["problem"]
    prob = resolve_problem(problem_name)
    println("Problem: $problem_name")

    # Generate test data
    n_test = args["n-test"]
    tc = get(ckpt, "training_config", Dict())
    sigma = get(tc, "sigma", 5.0)
    println("Generating $n_test test samples...")
    test_data = _generate_dataset_for_problem(prob, n_test; sigma=sigma)

    # Evaluate
    mu_prim = args["mu-prim"]
    mu_surr = get(tc, "mu_surr", 1.0)

    regrets = Float64[]
    rel_regrets = Float64[]

    println("Evaluating...")
    for (i, (x, actual_sc)) in enumerate(test_data)
        raw = generator(x)
        pred = _params_to_scenarios(prob, raw)
        r = decision_regret(prob, mu_surr, mu_prim, pred, actual_sc)
        push!(regrets, r)

        rr = relative_decision_regret(prob, mu_surr, mu_prim, pred, actual_sc)
        push!(rel_regrets, rr)
    end

    println("\n=== Test Results ($n_test samples) ===")
    println("Decision regret:")
    println("  Mean:    $(mean(regrets))")
    println("  Std:     $(std(regrets))")
    println("  Min:     $(minimum(regrets))")
    println("  Max:     $(maximum(regrets))")
    println("Relative decision regret:")
    println("  Mean:    $(mean(rel_regrets))")
    println("  Std:     $(std(rel_regrets))")
    println("  Min:     $(minimum(rel_regrets))")
    println("  Max:     $(maximum(rel_regrets))")

    # Save results if requested
    output_path = args["output"]
    if !isempty(output_path)
        open(output_path, "w") do io
            println(io, "sample,decision_regret,relative_decision_regret")
            for i in eachindex(regrets)
                println(io, "$i,$(regrets[i]),$(rel_regrets[i])")
            end
        end
        println("\nResults saved: $output_path")
    end

    return regrets
end

# -----------------------------------------------------------------------
# Subcommand: evaluate
# -----------------------------------------------------------------------

function cmd_evaluate(args::Dict)
    # Load checkpoint
    ckpt_path = args["checkpoint"]
    println("Loading checkpoint: $ckpt_path")
    ckpt = load_checkpoint(ckpt_path)

    # Seed
    seed = args["seed"]
    seed >= 0 && Random.seed!(seed)

    # Restore model
    generator = restore_model(ckpt)
    verbose = args["verbose"]
    verbose && println("Model restored: $(generator)")

    # Resolve problem
    problem_name = ckpt["problem_type"]
    prob = resolve_problem(problem_name)
    println("Problem: $problem_name")

    # Parameters
    tc = get(ckpt, "training_config", Dict())
    mu_surr = get(tc, "mu_surr", 1.0)
    mu_prim = args["mu-prim"]
    sigma = get(tc, "sigma", 5.0)

    # Generate test data
    n_test = args["n-test"]
    println("Generating $n_test test samples...")
    test_data = _generate_dataset_for_problem(prob, n_test; sigma=sigma)

    # Compute metrics
    println("Computing evaluation metrics...")
    metrics = compute_evaluation_metrics(generator, prob, test_data, mu_surr, mu_prim)

    # Print summary
    print_evaluation_summary(stdout, metrics)

    # Output directory and format
    output_dir = args["output-dir"]
    fmt = args["format"]

    # Plots
    if !args["no-plots"]
        mkpath(output_dir)
        println("\nGenerating plots in $output_dir/ ...")

        # Loss curve
        loss_history = get(ckpt, "loss_history", Float64[])
        if !isempty(loss_history)
            plot_loss_curve(loss_history, joinpath(output_dir, "loss_curve.$fmt"))
            verbose && println("  loss_curve.$fmt")
        end

        # Regret histogram
        rel_regrets = metrics["rel_regrets"]
        n = metrics["n"]
        ci = 1.96 * std(rel_regrets) / sqrt(n)
        plot_regret_histogram(rel_regrets, ci, joinpath(output_dir, "regret_histogram.$fmt"))
        verbose && println("  regret_histogram.$fmt")

        # Regret boxplot (violin)
        plot_regret_boxplot(rel_regrets, joinpath(output_dir, "regret_boxplot.$fmt"))
        verbose && println("  regret_boxplot.$fmt")

        # Regret CDF
        plot_regret_cdf(rel_regrets, joinpath(output_dir, "regret_cdf.$fmt"))
        verbose && println("  regret_cdf.$fmt")

        # Scenario scatter
        sc_dim = _scenario_param_dim(prob)
        if sc_dim <= 8
            plot_scenario_scatter(
                metrics["predicted_params"], metrics["actual_params"],
                joinpath(output_dir, "scenario_scatter.$fmt"))
            verbose && println("  scenario_scatter.$fmt")
        else
            verbose && println("  scenario_scatter skipped (dim=$sc_dim > 8)")
        end

        println("Plots saved to $output_dir/")
    end

    # CSV
    if args["csv"]
        mkpath(output_dir)
        csv_path = joinpath(output_dir, "metrics.csv")
        save_evaluation_csv(metrics, csv_path)
        println("Metrics CSV saved: $csv_path")
    end

    return metrics
end

# -----------------------------------------------------------------------
# Subcommand: info
# -----------------------------------------------------------------------

function cmd_info(args::Dict)
    ckpt_path = args["checkpoint"]
    ckpt = load_checkpoint(ckpt_path)
    println("File: $ckpt_path")
    println("Size: $(filesize(ckpt_path)) bytes")
    println()
    print_checkpoint_info(ckpt)
end

# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

"""
    cli_main()

Parse command-line arguments and dispatch to the appropriate subcommand handler.
"""
function cli_main()
    s = build_arg_parser()
    parsed = parse_args(ARGS, s)

    cmd = parsed["%COMMAND%"]

    if cmd == "train"
        cmd_train(parsed["train"])
    elseif cmd == "continue"
        cmd_continue(parsed["continue"])
    elseif cmd == "test"
        cmd_test(parsed["test"])
    elseif cmd == "evaluate"
        cmd_evaluate(parsed["evaluate"])
    elseif cmd == "info"
        cmd_info(parsed["info"])
    else
        _print_usage()
    end
end

function _print_usage()
    println("""
ProblemBasedScenarioGeneration CLI (pbsg)

Train, evaluate, and manage neural scenario generators for
two-stage stochastic linear programming.

COMMANDS
  train       Train a new scenario generator model
  continue    Resume training from a saved checkpoint
  test        Evaluate a trained model on test data (lightweight)
  evaluate    Full evaluation with metrics, plots, and CSV export
  info        Show checkpoint metadata and training history

PROBLEMS
  resource_allocation   20 clients, 30 resources (H_ONLY noise)
  shipment_planning     12 warehouses, 4 locations (H_ONLY noise)
  newsvendor            Unreliable newsvendor with demand + reliability (WH noise)

EXAMPLES
  # Train a newsvendor model
  julia bin/pbsg.jl train -p newsvendor --epochs 30 -v -o model.jls

  # Train with mu-continuation schedule
  julia bin/pbsg.jl train -p resource_allocation --continuation -v

  # Custom architecture
  julia bin/pbsg.jl train -p shipment_planning \\
      --hidden-dim 256 --n-layers 4 --lr 5e-4 --epochs 50 -v

  # Inspect a checkpoint
  julia bin/pbsg.jl info -c model.jls

  # Evaluate on test data (lightweight)
  julia bin/pbsg.jl test -c model.jls --n-test 100 -o results.csv

  # Full evaluation with plots and CSV
  julia bin/pbsg.jl evaluate -c model.jls --n-test 100 -d eval_out --csv

  # Resume training
  julia bin/pbsg.jl continue -c model.jls --epochs 10 --lr 1e-4 -v

Run 'julia bin/pbsg.jl <command> --help' for detailed options.""")
end
