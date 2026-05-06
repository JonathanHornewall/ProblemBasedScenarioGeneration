- check whether the problem remains well-defined no matter how you choose `h`.
    * reason : right now, we have cut off condition, so `h` becomes forced to be zero (if possible, because negative h constraint)
    * ask codex : whether it is possible that relaxing this requirement on the NN output could improve training quality ?
- Do 1000 scenario training run again, but this time, do NOT normalize the gradient step completely. Insetad, allow for
    * batch size 1 vs 3 on same epoch number and training data set
    * if 1000 data points
    * 




- on each problem, run five and benchmark against deterministic baselines
    * for each problem instances, make sure he implements it learning q, then learning h (two experiments)
- take the final mu of a run, replace it with rho (constant value) with mu at zero
