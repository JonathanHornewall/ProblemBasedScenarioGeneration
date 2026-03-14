using ProblemBasedScenarioGeneration
using Serialization

function save_trained_model(model, filepath)
    serialize(filepath, model)
end

function load_trained_model(filepath)
    deserialize(filepath)
end

function save_training_data(training_data, testing_data, filepath)
    serialize(filepath, (training_data, testing_data))
end

function load_training_data(filepath)
    deserialize(filepath)
end

function save_experiment_state(model, training_data, testing_data, problem_instance, reg_params;
                               filepath = "shipment_experiment_state.jls")
    pdata = problem_instance.problem_data
    problem_dict = Dict(
        "production_costs" => pdata.production_costs,
        "emergency_costs" => pdata.emergency_costs,
        "shipment_costs" => pdata.shipment_costs,
        "context_dimension" => pdata.context_dimension,
    )
    serialize(filepath, (model, training_data, testing_data, problem_dict, reg_params))
end

function load_experiment_state(filepath = "shipment_experiment_state.jls")
    model, training_data, testing_data, problem_dict, reg_params = deserialize(filepath)
    data = ShipmentPlanningProblemData(
        problem_dict["production_costs"],
        problem_dict["emergency_costs"],
        problem_dict["shipment_costs"];
        context_dimension = problem_dict["context_dimension"]
    )
    problem_instance = ShipmentPlanningProblem(data)
    return model, training_data, testing_data, problem_instance, reg_params
end
