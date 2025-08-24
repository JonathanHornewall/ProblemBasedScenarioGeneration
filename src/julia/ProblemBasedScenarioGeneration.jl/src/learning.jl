#=
# Initialise momentum 
opt_state = Flux.setup(Adam(0.01, 0.9), model)

for data in train_set
  grads = # grads with respect to w 
# call differentials_logbar_lp + Zygote  ChainRulesCore

# Update both model parameters and optimiser state:
Flux.update!(opt_state, model, grads[1])
end

using Flux
using Flux.Data: DataLoader
using Statistics        # for `mean`
=#


# ----------------------------------------------------------------------
# 1.  Loss on a mini‑batch
# ----------------------------------------------------------------------
"""
    batch_loss(model, problem_instance, reg_param_surr, reg_param_prim, ctx_batch, actual_scenarios)

Return the mean loss over all context variables contained in `ctx_batch`.
"""
batch_loss(model, problem_instance, reg_param_surr, reg_param_prim, ctx_batch, actual_scenarios) =
    mean(loss(problem_instance, reg_param_surr, reg_param_prim, model(ctx), actual_scenarios) for ctx in ctx_batch)


# ----------------------------------------------------------------------
# 2.  Main training routine
# ----------------------------------------------------------------------
"""
    train_model!(model, problem_instance, dataset;
                reg_param_surr, reg_param_prim, actual_scenarios, batchsize = 32, epochs = 50, lr = 1e-3, 
                optimiser       = ADAM)

Train `model` in place and return it.

* `problem_instance` — whatever your solvers need.
* `dataset`  — collection of context variables.
* `reg_param_surr`  — regularization parameter for surrogate problem.
* `reg_param_prim`  — regularization parameter for primal problem.
* `actual_scenarios` — actual scenario parameters for loss computation.
* `optimiser`        — any `Flux.Optimise.Optimiser` constructor
  (`ADAM`, `Descent`, `RMSProp`, …).
* `on_epoch_end`     — optional callback `(epoch, loss, model) -> nothing`.
"""
function train_model!(model, problem_instance, dataset, reg_param_surr, reg_param_prim, actual_scenarios,
                    batchsize = 32, epochs = 50, lr = 1e-3, optimiser = ADAM)

    opt = optimiser(lr)
    ps  = Flux.params(model)

    @info "Training for $epochs epochs (batchsize = $batchsize, reg_param_surr = $reg_param_surr, reg_param_prim = $reg_param_prim)"

    for epoch in 1:epochs
        for ctx_batch in DataLoader(dataset; batchsize=batchsize, shuffle=true)
            gs = gradient(ps) do
                batch_loss(model, problem_instance, reg_param_surr, reg_param_prim, ctx_batch, actual_scenarios)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end

        # --- monitoring ---------------------------------------------------
        sample_loss = batch_loss(model, problem_instance, reg_param_surr, reg_param_prim,
                                first(DataLoader(dataset; batchsize=1)), actual_scenarios)
        @info "epoch $epoch" train_loss = sample_loss
    end

    return model
end
