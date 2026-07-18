"""
Test suite for the ODE solver implementation.
"""

import numpy as np
import pytest
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


@pytest.fixture
def default_parameters():
    """Create default simulation parameters."""
    return SimulationParameters(
        reactor=ReactorParameters(
            volume=1.0,  # m³
            density=1000.0,  # kg/m³
            heat_capacity=4186.0,  # J/(kg·K)
            UA=5000.0,  # W/K
        ),
        feed=FeedParameters(
            flow_rate=0.001,  # m³/s
            concentration=1000.0,  # mol/m³
            temperature=298.15,  # K
        ),
        cooling=CoolingParameters(
            coolant_temperature=288.15,  # K
        ),
        kinetics=KineticParameters(
            k0=7.2e10,  # 1/s (pre-exponential factor)
            activation_energy=75000.0,  # J/mol
            heat_of_reaction=-50000.0,  # J/mol
            gas_constant=8.314,  # J/(mol·K)
        ),
    )


@pytest.fixture
def cstr_model(default_parameters):
    """Create a CSTR model."""
    return CSTR(default_parameters)


@pytest.fixture
def initial_state():
    """Create initial reactor state."""
    return ReactorState(concentration=100.0, temperature=300.0)


class TestODESolver:
    """Test cases for the ODE solver."""

    def test_solver_initialization(self, cstr_model):
        """Test ODESolver initialization with default method."""
        solver = ODESolver(cstr_model)
        assert solver.method == "RK45"
        assert solver.dense_output is True
        assert solver.model == cstr_model

    def test_solver_initialization_with_method(self, cstr_model):
        """Test ODESolver initialization with different methods."""
        for method in ["RK23", "BDF", "DOP853", "LSODA", "Radau"]:
            solver = ODESolver(cstr_model, method=method)
            assert solver.method == method

    def test_solver_initialization_invalid_method(self, cstr_model):
        """Test ODESolver initialization with invalid method."""
        with pytest.raises(ValueError, match="not recognized"):
            ODESolver(cstr_model, method="InvalidMethod")

    def test_solve_basic(self, cstr_model, initial_state):
        """Test basic ODE integration."""
        solver = ODESolver(cstr_model)
        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 1000.0),
            max_step=10.0,
        )

        # Check solution structure
        assert solution.t is not None
        assert solution.y is not None
        assert solution.sol is not None
        assert solution.status == 0  # Integration successful
        assert len(solution.t) > 0
        assert solution.y.shape[0] == 2  # Two state variables

    def test_solve_with_time_points(self, cstr_model, initial_state):
        """Test ODE integration with specified time points."""
        solver = ODESolver(cstr_model)
        t_eval = np.linspace(0.0, 1000.0, 101)
        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 1000.0),
            t_eval=t_eval,
        )

        # Check that solution matches requested time points
        assert len(solution.t) == len(t_eval)
        np.testing.assert_array_almost_equal(solution.t, t_eval)

    def test_solve_without_dense_output(self, cstr_model, initial_state):
        """Test ODE integration without dense output."""
        solver = ODESolver(cstr_model, dense_output=False)
        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 1000.0),
        )

        assert solution.sol is None

    def test_get_final_state(self, cstr_model, initial_state):
        """Test extraction of final state."""
        solver = ODESolver(cstr_model)
        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 100.0),
        )

        final_state = solver.get_final_state(solution)
        assert isinstance(final_state, ReactorState)
        assert final_state.concentration > 0
        assert final_state.temperature > 0

    def test_get_state_at_time(self, cstr_model, initial_state):
        """Test extraction of state at specific time."""
        solver = ODESolver(cstr_model)
        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 100.0),
        )

        # Get state at middle of integration
        t_mid = (solution.t[0] + solution.t[-1]) / 2
        state_at_mid = solver.get_state_at_time(solution, t_mid)
        assert isinstance(state_at_mid, ReactorState)

    def test_get_state_at_time_out_of_range(self, cstr_model, initial_state):
        """Test extraction of state at out-of-range time."""
        solver = ODESolver(cstr_model)
        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 100.0),
        )

        with pytest.raises(ValueError, match="outside solution range"):
            solver.get_state_at_time(solution, 1000.0)

    def test_evaluate_solution(self, cstr_model, initial_state):
        """Test evaluation of solution at arbitrary times."""
        solver = ODESolver(cstr_model)
        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 100.0),
        )

        # Evaluate at intermediate times
        t_new = np.array([10.0, 25.0, 50.0, 75.0, 99.0])
        y_new = solver.evaluate_solution(solution, t_new)

        assert y_new.shape == (2, len(t_new))
        assert np.all(np.isfinite(y_new))

    def test_evaluate_solution_without_dense_output(
        self, cstr_model, initial_state
    ):
        """Test that evaluation fails without dense output."""
        solver = ODESolver(cstr_model, dense_output=False)
        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 100.0),
        )

        t_new = np.array([10.0, 50.0])
        with pytest.raises(ValueError, match="does not have dense output"):
            solver.evaluate_solution(solution, t_new)

    def test_different_integration_methods(
        self, cstr_model, initial_state
    ):
        """Test integration with different solver methods."""
        methods = ["RK45", "RK23", "BDF", "LSODA"]
        
        for method in methods:
            solver = ODESolver(cstr_model, method=method)
            solution = solver.solve(
                initial_state=initial_state,
                t_span=(0.0, 100.0),
                max_step=5.0,
            )

            # All methods should produce valid solutions
            assert solution.status == 0
            assert len(solution.t) > 0
            assert solution.y.shape[0] == 2

    def test_solution_initial_conditions_preserved(
        self, cstr_model, initial_state
    ):
        """Test that initial conditions are correctly set."""
        solver = ODESolver(cstr_model)
        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 100.0),
            t_eval=np.linspace(0.0, 100.0, 11),
        )

        # Check first time point matches initial state
        initial_from_solution = solution.y[:, 0]
        expected_initial = initial_state.to_vector()
        np.testing.assert_array_almost_equal(
            initial_from_solution, expected_initial, decimal=5
        )

    def test_physical_constraints(self, cstr_model, initial_state):
        """Test that physical constraints are maintained."""
        solver = ODESolver(cstr_model)
        solution = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 1000.0),
        )

        # Concentration should remain non-negative
        assert np.all(solution.y[0, :] >= 0)
        # Temperature should remain above absolute zero
        assert np.all(solution.y[1, :] > 0)

    def test_time_tolerance_settings(self, cstr_model, initial_state):
        """Test integration with different tolerance settings."""
        solver = ODESolver(cstr_model)
        
        # Loose tolerances
        sol_loose = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 100.0),
            rtol=1e-2,
            atol=1e-5,
        )

        # Tight tolerances
        sol_tight = solver.solve(
            initial_state=initial_state,
            t_span=(0.0, 100.0),
            rtol=1e-6,
            atol=1e-9,
        )

        # Both should succeed and produce valid solutions
        assert sol_loose.status == 0
        assert sol_tight.status == 0
        # Tight tolerances should generally take more steps
        assert len(sol_tight.t) >= len(sol_loose.t)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
