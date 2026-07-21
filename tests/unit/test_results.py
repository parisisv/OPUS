import numpy as np
import pytest

from opus.core.results import SimulationResult


@pytest.fixture
def valid_result() -> SimulationResult:
    return SimulationResult(
        time=np.array([0.0, 1.0, 2.0]),
        states=np.array(
            [
                [10.0, 8.0, 6.0],
                [300.0, 310.0, 315.0],
            ]
        ),
        success=True,
        message="Simulation completed successfully.",
        nfev=42,
        njev=3,
        nlu=5,
    )


def test_result_dimensions(valid_result: SimulationResult) -> None:
    assert valid_result.number_of_states == 2
    assert valid_result.number_of_time_points == 3


def test_initial_state(valid_result: SimulationResult) -> None:
    np.testing.assert_allclose(
        valid_result.initial_state,
        np.array([10.0, 300.0]),
    )


def test_final_state(valid_result: SimulationResult) -> None:
    np.testing.assert_allclose(
        valid_result.final_state,
        np.array([6.0, 315.0]),
    )


def test_final_time(valid_result: SimulationResult) -> None:
    assert valid_result.final_time == pytest.approx(2.0)


def test_state_trajectory(valid_result: SimulationResult) -> None:
    trajectory = valid_result.state_trajectory(1)

    np.testing.assert_allclose(
        trajectory,
        np.array([300.0, 310.0, 315.0]),
    )


def test_result_converts_input_to_float64() -> None:
    result = SimulationResult(
        time=np.array([0, 1, 2]),
        states=np.array([[1, 2, 3]]),
        success=True,
        message="Success.",
        nfev=10,
    )

    assert result.time.dtype == np.float64
    assert result.states.dtype == np.float64


def test_result_arrays_are_read_only(
    valid_result: SimulationResult,
) -> None:
    with pytest.raises(ValueError):
        valid_result.time[0] = 5.0

    with pytest.raises(ValueError):
        valid_result.states[0, 0] = 5.0


def test_result_copies_input_arrays() -> None:
    time = np.array([0.0, 1.0])
    states = np.array([[1.0, 2.0]])

    result = SimulationResult(
        time=time,
        states=states,
        success=True,
        message="Success.",
        nfev=5,
    )

    time[0] = 100.0
    states[0, 0] = 100.0

    assert result.time[0] == pytest.approx(0.0)
    assert result.states[0, 0] == pytest.approx(1.0)


def test_rejects_non_one_dimensional_time() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        SimulationResult(
            time=np.array([[0.0, 1.0]]),
            states=np.array([[1.0, 2.0]]),
            success=True,
            message="Success.",
            nfev=5,
        )


def test_rejects_non_two_dimensional_states() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        SimulationResult(
            time=np.array([0.0, 1.0]),
            states=np.array([1.0, 2.0]),
            success=True,
            message="Success.",
            nfev=5,
        )


def test_rejects_inconsistent_time_and_state_dimensions() -> None:
    with pytest.raises(ValueError, match="columns"):
        SimulationResult(
            time=np.array([0.0, 1.0, 2.0]),
            states=np.array([[1.0, 2.0]]),
            success=True,
            message="Success.",
            nfev=5,
        )


def test_rejects_non_increasing_time() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        SimulationResult(
            time=np.array([0.0, 1.0, 1.0]),
            states=np.array([[1.0, 2.0, 3.0]]),
            success=True,
            message="Success.",
            nfev=5,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nfev", -1),
        ("njev", -1),
        ("nlu", -1),
    ],
)
def test_rejects_negative_solver_statistics(
    field: str,
    value: int,
) -> None:
    arguments = {
        "time": np.array([0.0, 1.0]),
        "states": np.array([[1.0, 2.0]]),
        "success": True,
        "message": "Success.",
        "nfev": 5,
        "njev": None,
        "nlu": None,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        SimulationResult(**arguments)


def test_rejects_invalid_state_index(
    valid_result: SimulationResult,
) -> None:
    with pytest.raises(IndexError, match="out of range"):
        valid_result.state_trajectory(2)