abstract type ProgramInstance end

stochastic_program(instance::ProgramInstance) =
    error("Stochastic-program construction is not defined for $(typeof(instance)).")

"""
    base_scenario(instance::ProgramInstance)

Return fixed scenario data for a program instance.

Concrete methods should return a `NamedTuple` with fields
`W_eq`, `W_ineq`, `T_eq`, `T_ineq`, `h_eq`, `h_ineq`, and `q`.
Scenario decoders can then combine this base scenario with generated
scenario parameters, usually by replacing one right-hand-side vector.
"""
base_scenario(instance::ProgramInstance) =
    error("Base-scenario construction is not defined for $(typeof(instance)).")
