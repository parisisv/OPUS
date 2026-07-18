"""
Example: Using the ODE Solver for CSTR Simulations

This example demonstrates how to use the ODESolver to simulate
a Continuous Stirred-Tank Reactor (CSTR) system.
"""

import numpy as np
import matplotlib.pyplot as plt

from opus.core.parameters import (
    SimulationParameters,
    ReactorParameters,
    FeedParameters,
    CoolingParameters,
    KineticParameters,
)
from opus.models.cstr import CSTR
from opus.models.state import ReactorState
from opus.solvers.ode_solver import ODESolver


def create_default_parameters():
    """Create default simulation parameters for the CSTR."""
    return SimulationParameters(
        reactor=ReactorParameters(
            volume=1.0,  # m³
            density=1000.0,  # kg/m³
            heat_capacity=4186.0,  # J/(kg·K)
            UA=5000.0,  # W/K
        ),
        feed=FeedParameters(
            flow_rate=0.001,  # m³/s (1 L/min)
            concentration=1000.0,  # mol/m³
            temperature=298.15,  # K (25°C)
        ),
        cooling=CoolingParameters(
            coolant_temperature=288.15,  # K (15°C)
        ),
        kinetics=KineticParameters(
            k0=7.2e10,  # 1/s (pre-exponential factor)
            activation_energy=75000.0,  # J/mol
            heat_of_reaction=-50000.0,  # J/mol (exothermic)
            gas_constant=8.314,  # J/(mol·K)
        ),
    )


def example_basic_simulation():
    """Example 1: Basic ODE integration."""
    print("=" * 70)
    print("Example 1: Basic ODE Integration")
    print("=" * 70)

    # Create model and solver
    params = create_default_parameters()
    model = CSTR(params)
    solver = ODESolver(model, method="RK45")

    # Define initial conditions
    initial_state = ReactorState(
        concentration=100.0,  # mol/m³
        temperature=300.0,  # K
    )

    print(f"Initial state:")
    print(f"  Concentration: {initial_state.concentration} mol/m³")
    print(f"  Temperature:   {initial_state.temperature} K")

    # Solve ODE
    solution = solver.solve(
        initial_state=initial_state,
        t_span=(0.0, 1000.0),  # seconds
        max_step=10.0,
    )

    # Extract final state
    final_state = solver.get_final_state(solution)
    print(f"\nFinal state (at t={solution.t[-1]:.1f}s):")
    print(f"  Concentration: {final_state.concentration:.2f} mol/m³")
    print(f"  Temperature:   {final_state.temperature:.2f} K")

    return solution, solver


def example_with_time_points(solution, solver):
    """Example 2: Integration with specified time points."""
    print("\n" + "=" * 70)
    print("Example 2: Integration with Specified Time Points")
    print("=" * 70)

    params = create_default_parameters()
    model = CSTR(params)
    solver = ODESolver(model)

    initial_state = ReactorState(
        concentration=100.0,
        temperature=300.0,
    )

    # Define time points of interest
    t_eval = np.linspace(0.0, 1000.0, 101)

    solution = solver.solve(
        initial_state=initial_state,
        t_span=(0.0, 1000.0),
        t_eval=t_eval,
    )

    print(f"Solution computed at {len(solution.t)} time points")
    print(f"Time range: {solution.t[0]:.1f} to {solution.t[-1]:.1f} seconds")

    # Access states at specific times
    states_at_times = [
        (0.0, solver.get_state_at_time(solution, 0.0)),
        (100.0, solver.get_state_at_time(solution, 100.0)),
        (500.0, solver.get_state_at_time(solution, 500.0)),
        (1000.0, solver.get_state_at_time(solution, 1000.0)),
    ]

    print("\nReactor state at selected times:")
    print(f"{'Time (s)':<12} {'Conc (mol/m³)':<16} {'Temp (K)':<12}")
    print("-" * 40)
    for t, state in states_at_times:
        print(
            f"{t:<12.1f} {state.concentration:<16.2f} "
            f"{state.temperature:<12.2f}"
        )

    return solution, solver


def example_different_methods():
    """Example 3: Comparing different integration methods."""
    print("\n" + "=" * 70)
    print("Example 3: Comparing Integration Methods")
    print("=" * 70)

    params = create_default_parameters()
    model = CSTR(params)

    initial_state = ReactorState(
        concentration=100.0,
        temperature=300.0,
    )

    methods = ["RK45", "RK23", "BDF", "LSODA"]
    results = {}

    print(f"{'Method':<10} {'N Steps':<10} {'Final C':<15} {'Final T':<12}")
    print("-" * 47)

    for method in methods:
        solver = ODESolver(model, method=method)
        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 500.0),
            max_step=5.0,
        )

        final_state = solver.get_final_state(solution)
        results[method] = solution

        print(
            f"{method:<10} {len(solution.t):<10} "
            f"{final_state.concentration:<15.2f} "
            f"{final_state.temperature:<12.2f}"
        )

    return results


def example_sensitivity_analysis():
    """Example 4: Sensitivity to initial conditions."""
    print("\n" + "=" * 70)
    print("Example 4: Sensitivity to Initial Conditions")
    print("=" * 70)

    params = create_default_parameters()
    model = CSTR(params)
    solver = ODESolver(model)

    # Test different initial concentrations
    initial_concentrations = [50.0, 100.0, 200.0]
    initial_temp = 300.0

    print(f"{'C_initial':<15} {'C_final':<15} {'T_final':<15}")
    print("-" * 45)

    for c_init in initial_concentrations:
        initial_state = ReactorState(
            concentration=c_init,
            temperature=initial_temp,
        )

        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 1000.0),
            max_step=10.0,
        )

        final_state = solver.get_final_state(solution)
        print(
            f"{c_init:<15.1f} {final_state.concentration:<15.2f} "
            f"{final_state.temperature:<15.2f}"
        )


def main():
    """Run all examples."""
    print("\nOPUS ODE Solver Examples\n")

    # Example 1: Basic simulation
    solution1, solver1 = example_basic_simulation()

    # Example 2: With time points
    solution2, solver2 = example_with_time_points(solution1, solver1)

    # Example 3: Different methods
    methods_results = example_different_methods()

    # Example 4: Sensitivity analysis
    example_sensitivity_analysis()

    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
