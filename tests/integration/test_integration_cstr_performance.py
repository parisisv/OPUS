import numpy as np
import pytest

from opus.analysis.cstr_performance import (
    SteadyStateTolerance,
    calculate_cstr_performance,
)
from opus.core.configuration import (
    SimulationConfiguration,
    SolverConfiguration,
    TimeConfiguration,
)
from opus.core.parameters import (
    CoolingParameters,
    FeedParameters,
    KineticParameters,
    ReactorParameters,
    SimulationParameters,
)
from opus.models.cstr import CSTR
from opus.models.state import ReactorState
from opus.solvers.ode_solver import ODESolver


def test_reproducible_cstr_simulation_and_performance_workflow() -> None:
    """
    Run a CSTR from a serializable configuration and calculate its
    engineering outputs.
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

    configuration = SimulationConfiguration(
        time=TimeConfiguration(
            initial_time=0.0,
            final_time=20_000.0,
            number_of_evaluation_points=2001,
        ),
        solver=SolverConfiguration(
            method="BDF",
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-11,
            max_step=10.0,
        ),
        initial_state=ReactorState(
            concentration=500.0,
            temperature=310.0,
        ),
    )

    # Confirm that the numerical experiment is serializable.
    serialized_configuration = configuration.model_dump(
        mode="json"
    )

    restored_configuration = (
        SimulationConfiguration.model_validate(
            serialized_configuration
        )
    )

    assert restored_configuration == configuration

    reactor = CSTR(parameters)

    solver = ODESolver(
        method=configuration.solver.method,
        relative_tolerance=(
            configuration.solver.relative_tolerance
        ),
        absolute_tolerance=(
            configuration.solver.absolute_tolerance
        ),
        max_step=configuration.solver.resolved_max_step,
    )

    result = solver.solve(
        model=reactor,
        initial_state=configuration.initial_state.to_vector(),
        time_span=configuration.time_span,
        evaluation_times=configuration.evaluation_times,
    )

    metrics = calculate_cstr_performance(
        model=reactor,
        result=result,
        steady_state_tolerance=SteadyStateTolerance(
            concentration=1.0e-7,
            temperature=1.0e-7,
        ),
    )

    assert result.success

    expected_residence_time = (
        parameters.reactor.volume
        / parameters.feed.flow_rate
    )

    assert metrics.residence_time == pytest.approx(
        expected_residence_time
    )

    expected_simulated_residence_times = (
        configuration.time.duration
        / expected_residence_time
    )

    assert metrics.simulated_residence_times == pytest.approx(
        expected_simulated_residence_times
    )

    assert metrics.final_concentration >= 0.0
    assert np.isfinite(metrics.final_conversion)

    assert metrics.minimum_temperature <= metrics.final_temperature
    assert metrics.maximum_temperature >= metrics.final_temperature

    expected_peak_temperature = float(
        np.max(result.states[1])
    )

    assert metrics.maximum_temperature == pytest.approx(
        expected_peak_temperature
    )

    expected_final_cooling_duty = (
        parameters.reactor.UA
        * (
            metrics.final_temperature
            - parameters.cooling.coolant_temperature
        )
    )

    assert metrics.final_cooling_duty == pytest.approx(
        expected_final_cooling_duty
    )

    final_derivative = reactor.rhs(
        result.final_time,
        result.final_state,
    )

    assert metrics.final_concentration_derivative == pytest.approx(
        final_derivative[0]
    )

    assert metrics.final_temperature_derivative == pytest.approx(
        final_derivative[1]
    )

    assert metrics.is_steady_state