using JuMP,
    Gurobi, 
    Distributions, 
    JLD2, 
    DataFrames,
    BilevelJuMP,
    Optim,
    NearestNeighbors,
    PyCall,
    StatsPlots,
    EmpiricalCDFs,
    Statistics,
    StatsBase,
    LinearAlgebra,
    Random,
    LinearRegression

import MathOptInterface
const MOI = MathOptInterface

const GRB_ENV = Gurobi.Env()

function set_solver_silent!(model::JuMP.Model)
    try
        set_optimizer_attribute(model, "OutputFlag", 0)
    catch err
        if err isa MOI.UnsupportedAttribute
            try
                JuMP.set_silent(model)
            catch
                # ignore
            end
        else
            rethrow()
        end
    end
end

