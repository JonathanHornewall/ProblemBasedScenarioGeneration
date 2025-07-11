# Initialise momentum 
opt_state = Flux.setup(Adam(0.01, 0.9), model)

for data in train_set
  grads = # grads with respect to w 
  # call differentials_logbar_lp + Zygote  ChainRulesCore

  # Update both model parameters and optimiser state:
  Flux.update!(opt_state, model, grads[1])
end