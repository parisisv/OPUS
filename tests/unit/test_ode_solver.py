import numpy as np
import pytest

from opus.core.interfaces import StateVector
from opus.solvers.ode_solver import ODESolver


class ExponentialDecayModel:
    """Simple model with a known analytical solution."""

    def __init__(self, rate_constant: float) -> None:
        self.rate_constant = rate_constant

    def rhs(
        self,
        t: float,
        y: StateVector,
    ) -> StateVector:
        del t
        return -self.rate_constant * y


@pytest.fixture
def decay_model() -> ExponentialDecayModel:
    return ExponentialDecayModel(rate_constant=0.5)


@pytest.fixture
def solver() -> ODESolver:
    return ODESolver(
        method="RK45",
        relative_tolerance=1.0e-9,
        absolute_tolerance=1.0e-12,
    )


def test_solver_returns_successful_result(
    solver: ODESolver,
    decay_model: ExponentialDecayModel,
) -> None:
    result = solver.solve(
        model=decay_model,
        initial_state=np.array([2.0]),
        time_span=(0.0, 5.0),
    )

    assert result.success
    assert result.nfev > 0
    assert result.number_of_states == 1
    assert result.final_time == pytest.approx(5.0)


def test_solver_matches_analytical_solution(
    solver: ODESolver,
    decay_model: ExponentialDecayModel,
) -> None:
    evaluation_times = np.linspace(0.0, 5.0, 21)

    result = solver.solve(
        model=decay_model,
        initial_state=np.array([2.0]),
        time_span=(0.0, 5.0),
        evaluation_times=evaluation_times,
    )

    expected = 2.0 * np.exp(-0.5 * evaluation_times)

    np.testing.assert_allclose(
        result.states[0],
        expected,
        rtol=1.0e-7,
        atol=1.0e-9,
    )


def test_solver_uses_requested_evaluation_times(
    solver: ODESolver,
    decay_model: ExponentialDecayModel,
) -> None:
    evaluation_times = np.array([0.0, 1.0, 2.0, 5.0])

    result = solver.solve(
        model=decay_model,
        initial_state=np.array([2.0]),
        time_span=(0.0, 5.0),
        evaluation_times=evaluation_times,
    )

    np.testing.assert_array_equal(
        result.time,
        evaluation_times,
    )


def test_solver_preserves_initial_state(
    solver: ODESolver,
    decay_model: ExponentialDecayModel,
) -> None:
    result = solver.solve(
        model=decay_model,
        initial_state=np.array([2.0]),
        time_span=(0.0, 1.0),
    )

    np.testing.assert_allclose(
        result.initial_state,
        np.array([2.0]),
    )


def test_solver_does_not_modify_initial_state(
    solver: ODESolver,
    decay_model: ExponentialDecayModel,
) -> None:
    initial_state = np.array([2.0])
    original_state = initial_state.copy()

    solver.solve(
        model=decay_model,
        initial_state=initial_state,
        time_span=(0.0, 1.0),
    )

    np.testing.assert_array_equal(
        initial_state,
        original_state,
    )


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
def test_solver_rejects_invalid_configuration(
    field: str,
    value: float,
) -> None:
    arguments = {
        "method": "RK45",
        "relative_tolerance": 1.0e-6,
        "absolute_tolerance": 1.0e-9,
        "max_step": np.inf,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        ODESolver(**arguments)


def test_solver_rejects_non_vector_initial_state(
    solver: ODESolver,
    decay_model: ExponentialDecayModel,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        solver.solve(
            model=decay_model,
            initial_state=np.array([[2.0]]),
            time_span=(0.0, 1.0),
        )


def test_solver_rejects_empty_initial_state(
    solver: ODESolver,
    decay_model: ExponentialDecayModel,
) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        solver.solve(
            model=decay_model,
            initial_state=np.array([]),
            time_span=(0.0, 1.0),
        )


def test_solver_rejects_non_finite_initial_state(
    solver: ODESolver,
    decay_model: ExponentialDecayModel,
) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        solver.solve(
            model=decay_model,
            initial_state=np.array([np.nan]),
            time_span=(0.0, 1.0),
        )


@pytest.mark.parametrize(
    "time_span",
    [
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_solver_rejects_invalid_time_span(
    solver: ODESolver,
    decay_model: ExponentialDecayModel,
    time_span: tuple[float, float],
) -> None:
    with pytest.raises(
        ValueError,
        match="final_time must be greater",
    ):
        solver.solve(
            model=decay_model,
            initial_state=np.array([2.0]),
            time_span=time_span,
        )


def test_solver_rejects_non_increasing_evaluation_times(
    solver: ODESolver,
    decay_model: ExponentialDecayModel,
) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        solver.solve(
            model=decay_model,
            initial_state=np.array([2.0]),
            time_span=(0.0, 5.0),
            evaluation_times=np.array([0.0, 2.0, 2.0, 5.0]),
        )


def test_solver_rejects_evaluation_times_outside_span(
    solver: ODESolver,
    decay_model: ExponentialDecayModel,
) -> None:
    with pytest.raises(ValueError, match="within time_span"):
        solver.solve(
            model=decay_model,
            initial_state=np.array([2.0]),
            time_span=(0.0, 5.0),
            evaluation_times=np.array([0.0, 2.0, 6.0]),
        )


def test_solver_rejects_object_without_rhs(
    solver: ODESolver,
) -> None:
    invalid_model = object()

    with pytest.raises(TypeError, match="DynamicModel"):
        solver.solve(
            model=invalid_model,  # type: ignore[arg-type]
            initial_state=np.array([2.0]),
            time_span=(0.0, 1.0),
        )