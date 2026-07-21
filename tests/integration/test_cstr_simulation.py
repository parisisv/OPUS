import numpy as np
from scipy.integrate import cumulative_trapezoid

from opus.core.parameters import (
    CoolingParameters,
    FeedParameters,
    KineticParameters,
    ReactorParameters,
    SimulationParameters,
)
from opus.models.cstr import CSTR
from opus.models.state import CSTRStateIndex, ReactorState
from opus.solvers.ode_solver import ODESolver


def test_cstr_satisfies_integrated_mass_and_energy_balances() -> None:
    """
    Verify the CSTR mass and energy balances over a dynamic simulation.

    The test independently reconstructs the balance right-hand sides and
    integrates them over time. The integrated changes must agree with the
    state changes reported by the ODE solver.
    """

    parameters = SimulationParameters(
        reactor=ReactorParameters(
            volume=1.0,
            density=1000.0,
            heat_capacity=4184.0,
            UA=500.0,
        ),
        feed=FeedParameters(
            flow_rate=1.0e-3,
            concentration=1000.0,
            temperature=300.0,
        ),
        cooling=CoolingParameters(
            coolant_temperature=295.0,
        ),
        kinetics=KineticParameters(
            k0=2.0e3,
            activation_energy=50_000.0,
            heat_of_reaction=-50_000.0,
            gas_constant=8.314462618,
        ),
    )

    reactor = CSTR(parameters)

    initial_state = ReactorState(
        concentration=500.0,
        temperature=310.0,
    )

    evaluation_times = np.linspace(
        0.0,
        5000.0,
        1001,
        dtype=np.float64,
    )

    solver = ODESolver(
        method="BDF",
        relative_tolerance=1.0e-9,
        absolute_tolerance=1.0e-11,
        max_step=10.0,
    )

    result = solver.solve(
        model=reactor,
        initial_state=initial_state.to_vector(),
        time_span=(evaluation_times[0], evaluation_times[-1]),
        evaluation_times=evaluation_times,
    )

    assert result.success

    concentration = result.states[
        CSTRStateIndex.CONCENTRATION
    ]
    temperature = result.states[
        CSTRStateIndex.TEMPERATURE
    ]

    reactor_parameters = parameters.reactor
    feed_parameters = parameters.feed
    cooling_parameters = parameters.cooling
    kinetic_parameters = parameters.kinetics

    # Independently reconstruct the Arrhenius rate constant and reaction rate.
    rate_constant = kinetic_parameters.k0 * np.exp(
        -kinetic_parameters.activation_energy
        / (
            kinetic_parameters.gas_constant
            * temperature
        )
    )

    reaction_rate = rate_constant * concentration

    # -------------------------------------------------------------------------
    # Mass balance
    #
    # dC_A/dt = (F/V)(C_A,f - C_A) - r_A
    # -------------------------------------------------------------------------

    mass_balance_rhs = (
        feed_parameters.flow_rate
        / reactor_parameters.volume
        * (
            feed_parameters.concentration
            - concentration
        )
        - reaction_rate
    )

    integrated_mass_change = cumulative_trapezoid(
        mass_balance_rhs,
        result.time,
        initial=0.0,
    )

    expected_concentration = (
        initial_state.concentration
        + integrated_mass_change
    )

    np.testing.assert_allclose(
        concentration,
        expected_concentration,
        rtol=2.0e-5,
        atol=2.0e-4,
    )

    # -------------------------------------------------------------------------
    # Energy balance
    #
    # dT/dt =
    #     (F/V)(T_f - T)
    #     + (-ΔH r_A)/(ρ Cp)
    #     - UA(T - T_c)/(ρ Cp V)
    # -------------------------------------------------------------------------

    feed_energy_term = (
        feed_parameters.flow_rate
        / reactor_parameters.volume
        * (
            feed_parameters.temperature
            - temperature
        )
    )

    reaction_energy_term = (
        -kinetic_parameters.heat_of_reaction
        * reaction_rate
        / (
            reactor_parameters.density
            * reactor_parameters.heat_capacity
        )
    )

    cooling_energy_term = (
        reactor_parameters.UA
        * (
            temperature
            - cooling_parameters.coolant_temperature
        )
        / (
            reactor_parameters.density
            * reactor_parameters.heat_capacity
            * reactor_parameters.volume
        )
    )

    energy_balance_rhs = (
        feed_energy_term
        + reaction_energy_term
        - cooling_energy_term
    )

    integrated_temperature_change = cumulative_trapezoid(
        energy_balance_rhs,
        result.time,
        initial=0.0,
    )

    expected_temperature = (
        initial_state.temperature
        + integrated_temperature_change
    )

    np.testing.assert_allclose(
        temperature,
        expected_temperature,
        rtol=2.0e-6,
        atol=2.0e-4,
    )
    
    
def test_cstr_approaches_steady_state() -> None:
    parameters = SimulationParameters(
        reactor=ReactorParameters(
            volume=1.0,
            density=1000.0,
            heat_capacity=4184.0,
            UA=500.0,
        ),
        feed=FeedParameters(
            flow_rate=1.0e-3,
            concentration=1000.0,
            temperature=300.0,
        ),
        cooling=CoolingParameters(
            coolant_temperature=295.0,
        ),
        kinetics=KineticParameters(
            k0=2.0e3,
            activation_energy=50_000.0,
            heat_of_reaction=-50_000.0,
            gas_constant=8.314462618,
        ),
    )

    reactor = CSTR(parameters)

    solver = ODESolver(
        method="BDF",
        relative_tolerance=1.0e-9,
        absolute_tolerance=1.0e-11,
        max_step=10.0,
    )

    result = solver.solve(
        model=reactor,
        initial_state=ReactorState(
            concentration=500.0,
            temperature=310.0,
        ).to_vector(),
        time_span=(0.0, 20_000.0),
    )

    final_derivative = reactor.rhs(
        result.final_time,
        result.final_state,
    )

    np.testing.assert_allclose(
        final_derivative,
        np.zeros(2),
        atol=1.0e-7,
    )