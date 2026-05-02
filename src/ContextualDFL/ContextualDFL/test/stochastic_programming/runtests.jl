import ChainRulesCore
import Serialization
import SparseArrays

function _captured_failure(f)
    try
        f()
    catch error
        return error
    end

    error("Expected stochastic program failure.")
end

function _crash_payload_from_failure(error)
    message = sprint(showerror, error)
    matched = match(r"Crash data serialized at (.+stochastic_program_failure\.jls)$", message)
    @test matched !== nothing
    crash_file = matched.captures[1]
    @test isfile(crash_file)
    return Serialization.deserialize(crash_file), crash_file, message
end

function _with_stochastic_crash_root(f)
    crash_root = mktempdir()
    previous_root = ContextualDFL._set_stochastic_crash_root!(crash_root)
    try
        return f(crash_root)
    finally
        ContextualDFL._set_stochastic_crash_root!(previous_root)
        rm(crash_root; recursive=true, force=true)
    end
end

@testset "stochastic_programming" begin
    solver = Solver(IpoptSolver(), HiGHSSolver())

    @testset "first-stage wrapper" begin
        first_stage_lp = LP(
            A_eq=zeros(0, 1),
            A_ineq=zeros(0, 1),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=[2.0],
        )
        program = StochasticProgram(first_stage_lp)

        @test program.first_stage_lp === first_stage_lp
        @test program.A_eq == first_stage_lp.A_eq
        @test program.A_ineq == first_stage_lp.A_ineq
        @test program.b_eq == first_stage_lp.b_eq
        @test program.b_ineq == first_stage_lp.b_ineq
        @test program.c == first_stage_lp.c
    end

    @testset "single-scenario solve crash serialization" begin
        _with_stochastic_crash_root() do crash_root
            program = StochasticProgram(
                A_eq=zeros(0, 0),
                A_ineq=zeros(0, 0),
                b_eq=Float64[],
                b_ineq=Float64[],
                c=Float64[],
            )
            W_eq_array = zeros(0, 1, 1)
            W_ineq_array = reshape([1.0, -1.0], 2, 1, 1)
            T_eq_array = zeros(0, 0, 1)
            T_ineq_array = zeros(2, 0, 1)
            h_eq_array = zeros(0, 1)
            h_ineq_array = reshape([0.0, -1.0], 2, 1)
            q_array = reshape([0.0], 1, 1)
            probabilities = [1.0]

            failure = _captured_failure() do
                solve(
                    solver,
                    program,
                    W_eq_array,
                    W_ineq_array,
                    T_eq_array,
                    T_ineq_array,
                    h_eq_array,
                    h_ineq_array,
                    q_array;
                    probabilities=probabilities,
                    μ=0,
                    constraint_tolerance=1e-7,
                )
            end

            payload, crash_file, message = _crash_payload_from_failure(failure)
            @test failure isa ContextualDFL.StochasticProgramFailure
            @test startswith(crash_file, crash_root)
            @test occursin("single-scenario problem failed.", message)
            @test payload.location === :single_scenario_solve
            @test payload.first_stage.A_eq == program.A_eq
            @test payload.first_stage.A_ineq == program.A_ineq
            @test payload.first_stage.b_eq == program.b_eq
            @test payload.first_stage.b_ineq == program.b_ineq
            @test payload.first_stage.c == program.c
            @test payload.scenario_data.W_eq_array == W_eq_array
            @test payload.scenario_data.W_ineq_array == W_ineq_array
            @test payload.scenario_data.T_eq_array == T_eq_array
            @test payload.scenario_data.T_ineq_array == T_ineq_array
            @test payload.scenario_data.h_eq_array == h_eq_array
            @test payload.scenario_data.h_ineq_array == h_ineq_array
            @test payload.scenario_data.q_array == q_array
            @test isempty(payload.scenario_data.W_eq_array)
            @test isempty(payload.scenario_data.h_eq_array)
            @test payload.μ == 0
            @test payload.effective_μ == zeros(2)
            @test payload.probabilities == probabilities
            @test payload.kwargs.constraint_tolerance == 1e-7
            @test payload.original_error_text != ""
        end
    end

    @testset "second-stage cost crash serialization" begin
        _with_stochastic_crash_root() do crash_root
            program = StochasticProgram(
                A_eq=zeros(0, 1),
                A_ineq=zeros(0, 1),
                b_eq=Float64[],
                b_ineq=Float64[],
                c=[0.0],
            )
            z = [0.0]
            W_eq_array = zeros(0, 1, 1)
            W_ineq_array = zeros(0, 1, 1)
            T_eq_array = zeros(0, 1, 1)
            T_ineq_array = zeros(0, 1, 1)
            h_eq_array = zeros(0, 1)
            h_ineq_array = zeros(0, 1)
            q_array = reshape([-1.0], 1, 1)

            failure = _captured_failure() do
                cost_function(
                    program,
                    solver,
                    z,
                    W_eq_array,
                    W_ineq_array,
                    T_eq_array,
                    T_ineq_array,
                    h_eq_array,
                    h_ineq_array,
                    q_array;
                    μ=0,
                    constraint_tolerance=1e-8,
                )
            end

            payload, crash_file, message = _crash_payload_from_failure(failure)
            @test failure isa ContextualDFL.StochasticProgramFailure
            @test startswith(crash_file, crash_root)
            @test occursin("second-stage problem failed in scenario 1.", message)
            @test payload.location === :second_stage_cost
            @test payload.scenario_index == 1
            @test payload.z == z
            @test payload.first_stage.c == program.c
            @test payload.scenario_data.W_eq_array == W_eq_array
            @test payload.scenario_data.W_ineq_array == W_ineq_array
            @test payload.scenario_data.T_eq_array == T_eq_array
            @test payload.scenario_data.T_ineq_array == T_ineq_array
            @test payload.scenario_data.h_eq_array == h_eq_array
            @test payload.scenario_data.h_ineq_array == h_ineq_array
            @test payload.scenario_data.q_array == q_array
            @test isempty(payload.scenario_data.W_ineq_array)
            @test isempty(payload.scenario_data.h_ineq_array)
            @test payload.μ == 0
            @test payload.scenario_μ == 0
            @test payload.kwargs.constraint_tolerance == 1e-8
            @test payload.original_error_text != ""
        end
    end

    @testset "two-scenario equality recourse" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=zeros(0, 1),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=[2.0],
        )

        W_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        W_ineq_array = zeros(0, 1, 2)
        T_eq_array = reshape([1.0, 2.0], 1, 1, 2)
        T_ineq_array = zeros(0, 1, 2)
        h_eq_array = reshape([5.0, 8.0], 1, 2)
        h_ineq_array = zeros(0, 2)
        q_array = reshape([3.0, 4.0], 1, 2)
        probabilities = [0.25, 0.75]
        z = [1.0]

        extensive_lp = construct_lp(
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        )

        @test extensive_lp.c == [2.0, 0.75, 3.0]
        @test SparseArrays.issparse(extensive_lp.A_eq)
        @test SparseArrays.issparse(extensive_lp.A_ineq)
        @test extensive_lp.A_eq == [1.0 1.0 0.0; 2.0 0.0 1.0]
        @test size(extensive_lp.A_ineq) == (0, 3)
        @test extensive_lp.b_eq == [5.0, 8.0]
        @test isempty(extensive_lp.b_ineq)

        @test cost_function(
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        ) ≈ 23.0 atol = 1e-8

        primal, dual_eq, dual_ineq = ContextualDFL.G_hat(
            solver,
            z,
            view(W_eq_array, :, :, 1),
            view(W_ineq_array, :, :, 1),
            view(T_eq_array, :, :, 1),
            view(T_ineq_array, :, :, 1),
            view(h_eq_array, :, 1),
            view(h_ineq_array, :, 1),
            view(q_array, :, 1);
            return_dual=true,
        )

        @test primal ≈ [4.0] atol = 1e-8
        @test dual_eq ≈ [3.0] atol = 1e-8
        @test isempty(dual_ineq)

        value, pullback = ChainRulesCore.rrule(
            cost_function,
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=0,
            probabilities=probabilities,
        )
        tangents = pullback(1.0)
        zero_tangents = pullback(ChainRulesCore.ZeroTangent())

        @test value ≈ 23.0 atol = 1e-8
        @test length(tangents) == 11
        @test tangents[4] ≈ [-4.75] atol = 1e-8
        @test zero_tangents[4] == zeros(size(z))
        @test_throws ArgumentError pullback([1.0])

        ϵ = 1e-5
        finite_difference_gradient = (
            cost_function(
                program,
                solver,
                z .+ ϵ,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                probabilities=probabilities,
            ) -
            cost_function(
                program,
                solver,
                z .- ϵ,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                probabilities=probabilities,
            )
        ) / (2ϵ)

        @test tangents[4][1] ≈ finite_difference_gradient atol = 1e-5
    end

    @testset "probability-scaled log barrier" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=reshape([-1.0, 1.0], 2, 1),
            b_eq=Float64[],
            b_ineq=[0.0, 10.0],
            c=[1.0],
        )

        W_eq_array = zeros(0, 1, 2)
        W_ineq_array = zeros(2, 1, 2)
        W_ineq_array[:, :, 1] = reshape([-1.0, 1.0], 2, 1)
        W_ineq_array[:, :, 2] = reshape([-1.0, 1.0], 2, 1)
        T_eq_array = zeros(0, 1, 2)
        T_ineq_array = zeros(2, 1, 2)
        h_eq_array = zeros(0, 2)
        h_ineq_array = [0.0 0.0; 5.0 7.0]
        q_array = reshape([2.0, 3.0], 1, 2)
        probabilities = [0.25, 0.75]
        μ = 0.4

        stochastic_result = solve(
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
            μ=μ,
            tol=1e-10,
        )

        extensive_lp = construct_lp(
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        )
        manual_barrier = [
            μ,
            μ,
            μ * probabilities[1],
            μ * probabilities[1],
            μ * probabilities[2],
            μ * probabilities[2],
        ]
        manual_result = solve(solver, extensive_lp; μ=manual_barrier, tol=1e-10)

        @test vcat(stochastic_result[1], vec(stochastic_result[2])) ≈ manual_result.z atol = 1e-8
    end

    @testset "log-barrier inequality dual convention" begin
        μ = 0.1
        lp = LP(
            A_eq=zeros(0, 1),
            A_ineq=reshape([-1.0, -1.0], 2, 1),
            b_eq=Float64[],
            b_ineq=[-8.0, 0.0],
            c=[1.0],
        )

        result = solve(solver, lp; μ=μ, tol=1e-10)
        slack = lp.b_ineq - lp.A_ineq * result.z

        @test minimum(result.dual_ineq) >= -1e-8
        @test result.dual_ineq ≈ fill(μ, length(slack)) ./ slack atol = 2e-4 rtol = 2e-4
    end

    @testset "log-barrier cost rrule inequality sign" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=zeros(0, 1),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=[0.0],
        )

        W_eq_array = zeros(0, 1, 1)
        W_ineq_array = reshape([-1.0, -1.0], 2, 1, 1)
        T_eq_array = zeros(0, 1, 1)
        T_ineq_array = reshape([-1.0, 0.0], 2, 1, 1)
        h_eq_array = zeros(0, 1)
        h_ineq_array = reshape([-10.0, 0.0], 2, 1)
        q_array = reshape([1.0], 1, 1)
        z = [2.0]
        μ = 0.1

        value, pullback = ChainRulesCore.rrule(
            cost_function,
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=μ,
            tol=1e-10,
        )
        dz = pullback(1.0)[4]

        direction = [0.2]
        ϵ = 1e-5
        finite_difference_gradient = (
            cost_function(
                program,
                solver,
                z .+ ϵ .* direction,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                μ=μ,
                tol=1e-10,
            ) -
            cost_function(
                program,
                solver,
                z .- ϵ .* direction,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array;
                μ=μ,
                tol=1e-10,
            )
        ) / (2ϵ)

        @test value isa Number
        @test sum(dz .* direction) ≈ finite_difference_gradient atol = 2e-4 rtol = 2e-4
    end

    @testset "single-scenario equality and inequality recourse" begin
        z = [1.0]
        W_eq = [1.0 0.0]
        W_ineq = [0.0 1.0]
        T_eq = reshape([1.0], 1, 1)
        T_ineq = reshape([-1.0], 1, 1)
        h_eq = [4.0]
        h_ineq = [3.0]
        q = [1.0, -2.0]

        @test ContextualDFL.G_hat(
            solver,
            z,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q,
        ) ≈ -5.0 atol = 1e-8

        primal, dual_eq, dual_ineq = ContextualDFL.G_hat(
            solver,
            z,
            W_eq,
            W_ineq,
            T_eq,
            T_ineq,
            h_eq,
            h_ineq,
            q;
            return_dual=true,
        )

        @test primal ≈ [3.0, 4.0] atol = 1e-8
        @test dual_eq ≈ [1.0] atol = 1e-8
        @test dual_ineq ≈ [2.0] atol = 1e-8
    end

    @testset "solve packages first and second stage solutions" begin
        program = StochasticProgram(
            A_eq=reshape([1.0], 1, 1),
            A_ineq=zeros(0, 1),
            b_eq=[1.0],
            b_ineq=Float64[],
            c=[0.0],
        )

        W_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        W_ineq_array = zeros(0, 1, 2)
        T_eq_array = reshape([1.0, 2.0], 1, 1, 2)
        T_ineq_array = zeros(0, 1, 2)
        h_eq_array = reshape([5.0, 8.0], 1, 2)
        h_ineq_array = zeros(0, 2)
        q_array = reshape([3.0, 4.0], 1, 2)

        z, y, λ_b_eq, λ_b_ineq, λ_h_eq_array, λ_h_ineq_array = solve(
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
        )

        @test z ≈ [1.0] atol = 1e-8
        @test y ≈ reshape([4.0, 6.0], 1, 2) atol = 1e-8
        @test length(λ_b_eq) == 1
        @test isempty(λ_b_ineq)
        @test size(λ_h_eq_array) == (1, 2)
        @test size(λ_h_ineq_array) == (0, 2)
    end

    @testset "solve optimizes first-stage recourse tradeoff" begin
        program = StochasticProgram(
            A_eq=zeros(0, 1),
            A_ineq=reshape([1.0, -1.0], 2, 1),
            b_eq=Float64[],
            b_ineq=[3.0, 0.0],
            c=[2.0],
        )

        W_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        W_ineq_array = zeros(0, 1, 2)
        T_eq_array = reshape([1.0, 1.0], 1, 1, 2)
        T_ineq_array = zeros(0, 1, 2)
        h_eq_array = reshape([4.0, 6.0], 1, 2)
        h_ineq_array = zeros(0, 2)
        q_array = reshape([3.0, 4.0], 1, 2)
        probabilities = [0.5, 0.5]

        z, y, _, _, _, _ = solve(
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        )

        @test z ≈ [3.0] atol = 1e-8
        @test y ≈ reshape([1.0, 3.0], 1, 2) atol = 1e-8
        @test cost_function(
            program,
            solver,
            z,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            probabilities=probabilities,
        ) ≈ 13.5 atol = 1e-8
    end

    @testset "solve rrule returns h and q tangents only" begin
        program = StochasticProgram(
            A_eq=reshape([1.0], 1, 1),
            A_ineq=zeros(0, 1),
            b_eq=[1.0],
            b_ineq=Float64[],
            c=[0.0],
        )

        W_eq_array = reshape([1.0], 1, 1, 1)
        W_ineq_array = zeros(0, 1, 1)
        T_eq_array = reshape([1.0], 1, 1, 1)
        T_ineq_array = zeros(0, 1, 1)
        h_eq_array = reshape([2.0], 1, 1)
        h_ineq_array = zeros(0, 1)
        q_array = reshape([3.0], 1, 1)

        output, pullback = ChainRulesCore.rrule(
            solve,
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=0,
        )
        output_tangent = (
            zeros(1),
            ones(1, 1),
            zeros(1),
            zeros(0),
            zeros(1, 1),
            zeros(0, 1),
        )
        tangents = pullback(output_tangent)
        structured_tangents = pullback(ChainRulesCore.Tangent{typeof(output)}(output_tangent...))

        @test output[1] ≈ [1.0] atol = 1e-8
        @test output[2] ≈ reshape([1.0], 1, 1) atol = 1e-8
        @test tangents[8] ≈ reshape([1.0], 1, 1) atol = 1e-8
        @test tangents[9] == zeros(0, 1)
        @test tangents[10] ≈ reshape([0.0], 1, 1) atol = 1e-8
        @test structured_tangents[8] ≈ tangents[8] atol = 1e-8
        @test structured_tangents[9] == tangents[9]
        @test structured_tangents[10] ≈ tangents[10] atol = 1e-8

        ϵ = 1e-5
        h_fd = (
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array .+ ϵ,
                h_ineq_array,
                q_array;
                μ=0,
            )[2][1, 1] -
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array .- ϵ,
                h_ineq_array,
                q_array;
                μ=0,
            )[2][1, 1]
        ) / (2ϵ)

        q_fd = (
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array .+ ϵ;
                μ=0,
            )[2][1, 1] -
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array .- ϵ;
                μ=0,
            )[2][1, 1]
        ) / (2ϵ)

        @test tangents[8][1, 1] ≈ h_fd atol = 1e-5
        @test tangents[10][1, 1] ≈ q_fd atol = 1e-5
    end

    @testset "solve rrule rejects dual output cotangents" begin
        program = StochasticProgram(
            A_eq=reshape([1.0], 1, 1),
            A_ineq=reshape([1.0], 1, 1),
            b_eq=[1.0],
            b_ineq=[2.0],
            c=[0.0],
        )

        W_eq_array = reshape([1.0], 1, 1, 1)
        W_ineq_array = reshape([1.0], 1, 1, 1)
        T_eq_array = reshape([1.0], 1, 1, 1)
        T_ineq_array = reshape([0.0], 1, 1, 1)
        h_eq_array = reshape([2.0], 1, 1)
        h_ineq_array = reshape([3.0], 1, 1)
        q_array = reshape([1.0], 1, 1)

        output, pullback = ChainRulesCore.rrule(
            solve,
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=0,
        )
        zero_output_tangent = map(x -> zeros(size(x)), output)
        zero_tangents = pullback(zero_output_tangent)

        @test zero_tangents[8] == zeros(size(h_eq_array))
        @test zero_tangents[9] == zeros(size(h_ineq_array))
        @test zero_tangents[10] == zeros(size(q_array))

        for index in 3:6
            component_tangent = copy(zero_output_tangent[index])
            component_tangent[firstindex(component_tangent)] = 1.0
            output_tangent = ntuple(
                i -> i == index ? component_tangent : zero_output_tangent[i],
                length(zero_output_tangent),
            )

            @test_throws ArgumentError pullback(output_tangent)
        end
    end

    @testset "log-barrier solve rrule q sensitivity" begin
        program = StochasticProgram(
            A_eq=zeros(0, 0),
            A_ineq=zeros(0, 0),
            b_eq=Float64[],
            b_ineq=Float64[],
            c=Float64[],
        )

        W_eq_array = zeros(0, 1, 1)
        W_ineq_array = reshape([1.0, -1.0], 2, 1, 1)
        T_eq_array = zeros(0, 0, 1)
        T_ineq_array = zeros(2, 0, 1)
        h_eq_array = zeros(0, 1)
        h_ineq_array = reshape([1.0, 1.0], 2, 1)
        q_array = reshape([0.0], 1, 1)

        output, pullback = ChainRulesCore.rrule(
            solve,
            solver,
            program,
            W_eq_array,
            W_ineq_array,
            T_eq_array,
            T_ineq_array,
            h_eq_array,
            h_ineq_array,
            q_array;
            μ=0.5,
        )
        output_tangent = (
            Float64[],
            ones(1, 1),
            Float64[],
            Float64[],
            zeros(0, 1),
            zeros(2, 1),
        )
        tangents = pullback(output_tangent)

        @test output[2] ≈ reshape([0.0], 1, 1) atol = 1e-5
        @test tangents[10] ≈ reshape([-1.0], 1, 1) atol = 1e-4

        ϵ = 1e-2
        q_fd = (
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array .+ ϵ;
                μ=0.5,
                tol=1e-10,
            )[2][1, 1] -
            solve(
                solver,
                program,
                W_eq_array,
                W_ineq_array,
                T_eq_array,
                T_ineq_array,
                h_eq_array,
                h_ineq_array,
                q_array .- ϵ;
                μ=0.5,
                tol=1e-10,
            )[2][1, 1]
        ) / (2ϵ)

        @test tangents[10][1, 1] ≈ q_fd atol = 1e-3
    end
end
