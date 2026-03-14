"""
μ-continuation training schedule (Algorithm 1 from the paper).

Trains a scenario generator by gradually annealing the barrier parameter μ
from a large value (smooth landscape) to near-zero (close to true decision regret).
"""

"""
    continuation_train!(model, prob, dataset;
        mu_schedule=[1.0, 0.8, ...],
        epochs_per_stage=10,
        first_stage_epochs=20,
        finetune_epochs=10,
        opt=Adam(1e-3),
        verbose=false) -> Vector{Float64}

Train a scenario generator using the μ-continuation schedule.

Phases:
1. **Warm-up** (first_stage_epochs): train at `mu_schedule[1]` (large μ).
2. **Annealing** (epochs_per_stage per stage): step through `mu_schedule`.
3. **Fine-tuning** (finetune_epochs): train at μ=0 (true decision regret).

Returns the concatenated loss history across all phases.
"""
function continuation_train!(
    model,
    prob::ProblemInstance,
    dataset::Vector{<:Tuple};
    mu_schedule::Vector{Float64} = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01],
    epochs_per_stage::Int  = 10,
    first_stage_epochs::Int = 20,
    finetune_epochs::Int    = 10,
    opt                     = Adam(1e-3),
    verbose::Bool           = false
)
    all_losses = Float64[]

    # Phase 1: warm-up at initial μ
    mu0 = mu_schedule[1]
    verbose && @info "Continuation phase 1: warm-up at μ=$mu0 for $first_stage_epochs epochs"
    losses = train!(model, prob, dataset;
                    mu_surr=mu0, mu_prim=mu0, opt=opt,
                    epochs=first_stage_epochs, verbose=verbose)
    append!(all_losses, losses)

    # Phase 2: annealing
    for (stage, mu) in enumerate(mu_schedule)
        verbose && @info "Continuation stage $stage: μ=$mu for $epochs_per_stage epochs"
        losses = train!(model, prob, dataset;
                        mu_surr=mu, mu_prim=mu, opt=opt,
                        epochs=epochs_per_stage, verbose=verbose)
        append!(all_losses, losses)
    end

    # Phase 3: fine-tuning at μ=0
    verbose && @info "Continuation fine-tuning: μ=0 for $finetune_epochs epochs"
    losses = train!(model, prob, dataset;
                    mu_surr=0.0, mu_prim=0.0, opt=opt,
                    epochs=finetune_epochs, verbose=verbose)
    append!(all_losses, losses)

    return all_losses
end
