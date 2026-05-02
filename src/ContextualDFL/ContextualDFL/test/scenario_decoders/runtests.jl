import ChainRulesCore
import Flux

struct TestScenarioDecoder <: ScenarioDecoder end

struct TestVectorDecoder <: VectorDecoder end

function (::TestVectorDecoder)(vector::AbstractVector)
    return (
        reshape(view(vector, 1:1), 1, 1),
        zeros(eltype(vector), 0, 1),
        reshape(view(vector, 2:2), 1, 1),
        zeros(eltype(vector), 0, 1),
        view(vector, 3:3),
        zeros(eltype(vector), 0),
        view(vector, 4:4),
    )
end

@testset "scenario_decoders" begin
    @test_throws ErrorException TestScenarioDecoder()(:ξ)
    @test_throws ErrorException TestVectorDecoder()(:bad)

    scenario_parameters = ParametricScenario(;
        W_eq_xi=:parameter_W_eq,
        W_ineq_xi=:parameter_W_ineq,
        T_eq_xi=:parameter_T_eq,
        T_ineq_xi=:parameter_T_ineq,
        h_eq_xi=:parameter_h_eq,
        h_ineq_xi=:parameter_h_ineq,
        q_xi=:parameter_q,
    )

    decoder = ParametricDecoder(
        (:W_eq, :h_eq);
        base_W_eq=:base_W_eq,
        base_W_ineq=:base_W_ineq,
        base_T_eq=:base_T_eq,
        base_T_ineq=:base_T_ineq,
        base_h_eq=:base_h_eq,
        base_h_ineq=:base_h_ineq,
        base_q=:base_q,
    )
    W_eq, W_ineq, T_eq, T_ineq, h_eq, h_ineq, q = decoder(scenario_parameters)

    @test W_eq === :parameter_W_eq
    @test h_eq === :parameter_h_eq
    @test W_ineq === :base_W_ineq
    @test T_eq === :base_T_eq
    @test T_ineq === :base_T_ineq
    @test h_ineq === :base_h_ineq
    @test q === :base_q

    @test ParametricDecoder(
        (:T_eq,);
        base_W_eq=:base_W_eq,
        base_W_ineq=:base_W_ineq,
        base_T_ineq=:base_T_ineq,
        base_h_eq=:base_h_eq,
        base_h_ineq=:base_h_ineq,
        base_q=:base_q,
    )(scenario_parameters)[3] === :parameter_T_eq
    @test_throws ArgumentError ParametricDecoder((:bad_component,))
    @test_throws ArgumentError ParametricDecoder((:q,))(scenario_parameters)

    scenario_collection = [
        ParametricScenario(;
            W_eq_xi=[1.0 2.0; 3.0 4.0],
            W_ineq_xi=[5.0 6.0],
            T_eq_xi=reshape([7.0, 8.0], 2, 1),
            T_ineq_xi=reshape([9.0], 1, 1),
            h_eq_xi=[10.0, 11.0],
            h_ineq_xi=[12.0],
            q_xi=[13.0, 14.0],
        ),
        ParametricScenario(;
            W_eq_xi=[15.0 16.0; 17.0 18.0],
            W_ineq_xi=[19.0 20.0],
            T_eq_xi=reshape([21.0, 22.0], 2, 1),
            T_ineq_xi=reshape([23.0], 1, 1),
            h_eq_xi=[24.0, 25.0],
            h_ineq_xi=[26.0],
            q_xi=[27.0, 28.0],
        ),
    ]
    collection_decoder =
        ParametricDecoder((:W_eq, :W_ineq, :T_eq, :T_ineq, :h_eq, :h_ineq, :q))

    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array = decode_scenario_collection(collection_decoder, scenario_collection)

    @test W_eq_array[:, :, 1] == scenario_collection[1].W_eq_xi
    @test W_eq_array[:, :, 2] == scenario_collection[2].W_eq_xi
    @test W_ineq_array[:, :, 1] == scenario_collection[1].W_ineq_xi
    @test W_ineq_array[:, :, 2] == scenario_collection[2].W_ineq_xi
    @test T_eq_array[:, :, 1] == scenario_collection[1].T_eq_xi
    @test T_eq_array[:, :, 2] == scenario_collection[2].T_eq_xi
    @test T_ineq_array[:, :, 1] == scenario_collection[1].T_ineq_xi
    @test T_ineq_array[:, :, 2] == scenario_collection[2].T_ineq_xi
    @test h_eq_array[:, 1] == scenario_collection[1].h_eq_xi
    @test h_eq_array[:, 2] == scenario_collection[2].h_eq_xi
    @test h_ineq_array[:, 1] == scenario_collection[1].h_ineq_xi
    @test h_ineq_array[:, 2] == scenario_collection[2].h_ineq_xi
    @test q_array[:, 1] == scenario_collection[1].q_xi
    @test q_array[:, 2] == scenario_collection[2].q_xi

    _, pullback =
        ChainRulesCore.rrule(decode_scenario_collection, collection_decoder, scenario_collection)
    collection_tangent = pullback((
        fill(1.0, size(W_eq_array)),
        fill(2.0, size(W_ineq_array)),
        fill(3.0, size(T_eq_array)),
        fill(4.0, size(T_ineq_array)),
        fill(5.0, size(h_eq_array)),
        fill(6.0, size(h_ineq_array)),
        fill(7.0, size(q_array)),
    ))[3]

    @test collection_tangent[1].W_eq_xi == fill(1.0, size(scenario_collection[1].W_eq_xi))
    @test collection_tangent[2].W_ineq_xi == fill(2.0, size(scenario_collection[2].W_ineq_xi))
    @test collection_tangent[1].T_eq_xi == fill(3.0, size(scenario_collection[1].T_eq_xi))
    @test collection_tangent[2].T_ineq_xi == fill(4.0, size(scenario_collection[2].T_ineq_xi))
    @test collection_tangent[1].h_eq_xi == fill(5.0, size(scenario_collection[1].h_eq_xi))
    @test collection_tangent[2].h_ineq_xi == fill(6.0, size(scenario_collection[2].h_ineq_xi))
    @test collection_tangent[1].q_xi == fill(7.0, size(scenario_collection[1].q_xi))

    function zygote_collection_sum(x)
        W_eq_array,
        W_ineq_array,
        T_eq_array,
        T_ineq_array,
        h_eq_array,
        h_ineq_array,
        q_array = decode_scenario_collection(
            collection_decoder,
            [
                ParametricScenario(;
                    W_eq_xi=reshape([x], 1, 1),
                    W_ineq_xi=reshape([2x], 1, 1),
                    T_eq_xi=reshape([3x], 1, 1),
                    T_ineq_xi=reshape([4x], 1, 1),
                    h_eq_xi=[5x],
                    h_ineq_xi=[6x],
                    q_xi=[7x],
                ),
            ],
        )

        return only(W_eq_array) +
               only(W_ineq_array) +
               only(T_eq_array) +
               only(T_ineq_array) +
               only(h_eq_array) +
               only(h_ineq_array) +
               only(q_array)
    end

    @test only(Flux.gradient(zygote_collection_sum, 2.0)) == 28.0
    @test_throws ArgumentError decode_scenario_collection(collection_decoder, ParametricScenario[])

    vector_arrays = decode_scenario_collection(TestVectorDecoder(), collect(1.0:8.0); nr_scenarios=2)
    @test size(vector_arrays[1]) == (1, 1, 2)
    @test vector_arrays[1][:, :, 1] == reshape([1.0], 1, 1)
    @test vector_arrays[1][:, :, 2] == reshape([5.0], 1, 1)
    @test vector_arrays[5] == reshape([3.0, 7.0], 1, 2)
    @test_throws ArgumentError decode_scenario_collection(TestVectorDecoder(), collect(1.0:5.0); nr_scenarios=2)
end
