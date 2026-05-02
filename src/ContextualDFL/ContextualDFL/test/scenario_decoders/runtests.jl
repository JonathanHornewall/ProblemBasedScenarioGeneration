import ChainRulesCore
import Zygote

struct TestScenarioDecoder <: ScenarioDecoder end

@testset "scenario_decoders" begin
    @test_throws ErrorException TestScenarioDecoder()(:ξ)

    scenario_parameters = (;
        W_eq=:parameter_W_eq,
        W_ineq=:parameter_W_ineq,
        T_eq=:parameter_T_eq,
        T_ineq=:parameter_T_ineq,
        h_eq=:parameter_h_eq,
        h_ineq=:parameter_h_ineq,
        q=:parameter_q,
    )

    decoder = ComponentWiseDecoder(
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

    @test ComponentWiseDecoder(
        (:T_eq,);
        base_W_eq=:base_W_eq,
        base_W_ineq=:base_W_ineq,
        base_T_ineq=:base_T_ineq,
        base_h_eq=:base_h_eq,
        base_h_ineq=:base_h_ineq,
        base_q=:base_q,
    )(scenario_parameters)[3] === :parameter_T_eq
    @test_throws ArgumentError ComponentWiseDecoder((:bad_component,))
    @test_throws ArgumentError ComponentWiseDecoder((:q,))(scenario_parameters)

    scenario_collection = [
        (;
            W_eq=[1.0 2.0; 3.0 4.0],
            W_ineq=[5.0 6.0],
            T_eq=reshape([7.0, 8.0], 2, 1),
            T_ineq=reshape([9.0], 1, 1),
            h_eq=[10.0, 11.0],
            h_ineq=[12.0],
            q=[13.0, 14.0],
        ),
        (;
            W_eq=[15.0 16.0; 17.0 18.0],
            W_ineq=[19.0 20.0],
            T_eq=reshape([21.0, 22.0], 2, 1),
            T_ineq=reshape([23.0], 1, 1),
            h_eq=[24.0, 25.0],
            h_ineq=[26.0],
            q=[27.0, 28.0],
        ),
    ]
    collection_decoder =
        ComponentWiseDecoder((:W_eq, :W_ineq, :T_eq, :T_ineq, :h_eq, :h_ineq, :q))

    W_eq_array,
    W_ineq_array,
    T_eq_array,
    T_ineq_array,
    h_eq_array,
    h_ineq_array,
    q_array = decode_scenario_collection(collection_decoder, scenario_collection)

    @test W_eq_array[:, :, 1] == scenario_collection[1].W_eq
    @test W_eq_array[:, :, 2] == scenario_collection[2].W_eq
    @test W_ineq_array[:, :, 1] == scenario_collection[1].W_ineq
    @test W_ineq_array[:, :, 2] == scenario_collection[2].W_ineq
    @test T_eq_array[:, :, 1] == scenario_collection[1].T_eq
    @test T_eq_array[:, :, 2] == scenario_collection[2].T_eq
    @test T_ineq_array[:, :, 1] == scenario_collection[1].T_ineq
    @test T_ineq_array[:, :, 2] == scenario_collection[2].T_ineq
    @test h_eq_array[:, 1] == scenario_collection[1].h_eq
    @test h_eq_array[:, 2] == scenario_collection[2].h_eq
    @test h_ineq_array[:, 1] == scenario_collection[1].h_ineq
    @test h_ineq_array[:, 2] == scenario_collection[2].h_ineq
    @test q_array[:, 1] == scenario_collection[1].q
    @test q_array[:, 2] == scenario_collection[2].q

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

    @test collection_tangent[1].W_eq == fill(1.0, size(scenario_collection[1].W_eq))
    @test collection_tangent[2].W_ineq == fill(2.0, size(scenario_collection[2].W_ineq))
    @test collection_tangent[1].T_eq == fill(3.0, size(scenario_collection[1].T_eq))
    @test collection_tangent[2].T_ineq == fill(4.0, size(scenario_collection[2].T_ineq))
    @test collection_tangent[1].h_eq == fill(5.0, size(scenario_collection[1].h_eq))
    @test collection_tangent[2].h_ineq == fill(6.0, size(scenario_collection[2].h_ineq))
    @test collection_tangent[1].q == fill(7.0, size(scenario_collection[1].q))

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
                (;
                    W_eq=reshape([x], 1, 1),
                    W_ineq=reshape([2x], 1, 1),
                    T_eq=reshape([3x], 1, 1),
                    T_ineq=reshape([4x], 1, 1),
                    h_eq=[5x],
                    h_ineq=[6x],
                    q=[7x],
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

    @test only(Zygote.gradient(zygote_collection_sum, 2.0)) == 28.0
    @test_throws ArgumentError decode_scenario_collection(collection_decoder, NamedTuple[])
end
