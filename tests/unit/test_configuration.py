import numpy as np
import pytest
from pydantic import ValidationError

from opus.core.configuration import (
    SimulationConfiguration,
    SolverConfiguration,
    TimeConfiguration,
)
from opus.models.state import ReactorState


def test_default_solver_configuration() -> None:
    configuration = SolverConfiguration()

    assert configuration.method == "RK45"
    assert configuration.relative_tolerance == pytest.approx(1.0e-6)
    assert configuration.absolute_tolerance == pytest.approx(1.0e-9)
    assert configuration.max_step is None
    assert np.isinf(configuration.resolved_max_step)


def test_solver_configuration_resolves_max_step() -> None:
    configuration = SolverConfiguration(max_step=5.0)

    assert configuration.resolved_max_step == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relative_tolerance", 0.0),
        ("relative_tolerance", -1.0e-6),
        ("absolute_tolerance", 0.0),
        ("absolute_tolerance", -1.0e-9),
        ("max_step", 0.0),
        ("max_step", -1.0),
    ],
)
def test_rejects_invalid_solver_configuration(
    field: str,
    value: float,
) -> None:
    arguments = {
        "relative_tolerance": 1.0e-6,
        "absolute_tolerance": 1.0e-9,
        "max_step": None,
    }
    arguments[field] = value

    with pytest.raises(ValidationError):
        SolverConfiguration(**arguments)


def test_time_configuration_properties() -> None:
    configuration = TimeConfiguration(
        initial_time=2.0,
        final_time=12.0,
        number_of_evaluation_points=6,
    )

    assert configuration.time_span == (2.0, 12.0)
    assert configuration.duration == pytest.approx(10.0)


def test_time_configuration_generates_evaluation_grid() -> None:
    configuration = TimeConfiguration(
        initial_time=0.0,
        final_time=10.0,
        number_of_evaluation_points=6,
    )

    expected = np.array(
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    )

    evaluation_times = configuration.evaluation_times

    assert evaluation_times is not None
    np.testing.assert_allclose(evaluation_times, expected)
    assert not evaluation_times.flags.writeable


def test_time_configuration_without_output_grid() -> None:
    configuration = TimeConfiguration(
        initial_time=0.0,
        final_time=10.0,
    )

    assert configuration.evaluation_times is None


@pytest.mark.parametrize(
    ("initial_time", "final_time"),
    [
        (1.0, 1.0),
        (2.0, 1.0),
        (np.nan, 1.0),
        (0.0, np.inf),
    ],
)
def test_rejects_invalid_time_interval(
    initial_time: float,
    final_time: float,
) -> None:
    with pytest.raises(ValidationError):
        TimeConfiguration(
            initial_time=initial_time,
            final_time=final_time,
        )


@pytest.mark.parametrize(
    "number_of_evaluation_points",
    [0, 1, -1],
)
def test_rejects_invalid_evaluation_point_count(
    number_of_evaluation_points: int,
) -> None:
    with pytest.raises(ValidationError):
        TimeConfiguration(
            initial_time=0.0,
            final_time=10.0,
            number_of_evaluation_points=number_of_evaluation_points,
        )


def test_complete_simulation_configuration() -> None:
    configuration = SimulationConfiguration(
        time=TimeConfiguration(
            initial_time=0.0,
            final_time=100.0,
            number_of_evaluation_points=11,
        ),
        solver=SolverConfiguration(
            method="BDF",
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
            max_step=2.0,
        ),
        initial_state=ReactorState(
            concentration=500.0,
            temperature=310.0,
        ),
    )

    assert configuration.time_span == (0.0, 100.0)
    assert configuration.initial_state.concentration == pytest.approx(
        500.0
    )

    assert configuration.evaluation_times is not None
    assert configuration.evaluation_times.size == 11


def test_configuration_serialization_round_trip() -> None:
    original = SimulationConfiguration(
        time=TimeConfiguration(
            initial_time=0.0,
            final_time=100.0,
            number_of_evaluation_points=11,
        ),
        solver=SolverConfiguration(
            method="BDF",
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
            max_step=2.0,
        ),
        initial_state=ReactorState(
            concentration=500.0,
            temperature=310.0,
        ),
    )

    serialized = original.model_dump(mode="json")
    restored = SimulationConfiguration.model_validate(serialized)

    assert restored == original


def test_configuration_is_immutable() -> None:
    configuration = TimeConfiguration(
        initial_time=0.0,
        final_time=10.0,
    )

    with pytest.raises(ValidationError):
        configuration.final_time = 20.0