"""
Generic solver for ordinary differential equation models.

This module adapts SciPy's ``solve_ivp`` function to OPUS interfaces and
converts SciPy-specific outputs into ``SimulationResult`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from opus.core.interfaces import DynamicModel, StateVector
from opus.core.results import SimulationResult


SolverMethod = Literal[
    "RK23",
    "RK45",
    "DOP853",
    "Radau",
    "BDF",
    "LSODA",
]

TimeSpan = tuple[float, float]
TimeVector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ODESolver:
    """
    Generic solver for dynamic models governed by ordinary differential equations.

    Parameters
    ----------
    method
        Numerical integration method passed to SciPy's ``solve_ivp``.

    relative_tolerance
        Relative integration tolerance.

    absolute_tolerance
        Absolute integration tolerance. A scalar tolerance is applied to every
        state variable.

    max_step
        Maximum allowed integration step. The default permits the solver to
        choose its own step size without an explicit upper bound.
    """

    method: SolverMethod = "RK45"
    relative_tolerance: float = 1.0e-6
    absolute_tolerance: float = 1.0e-9
    max_step: float = np.inf

    def __post_init__(self) -> None:
        """Validate the solver configuration."""

        if self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be greater than zero.")

        if self.absolute_tolerance <= 0.0:
            raise ValueError("absolute_tolerance must be greater than zero.")

        if self.max_step <= 0.0:
            raise ValueError("max_step must be greater than zero.")

    def solve(
        self,
        model: DynamicModel,
        initial_state: StateVector,
        time_span: TimeSpan,
        *,
        evaluation_times: TimeVector | None = None,
    ) -> SimulationResult:
        """
        Integrate a dynamic model over a specified time interval.

        Parameters
        ----------
        model
            Dynamic model implementing the ``DynamicModel`` protocol.

        initial_state
            One-dimensional NumPy array containing the initial state values.

        time_span
            Tuple containing the initial and final simulation times.

        evaluation_times
            Optional one-dimensional array of times at which the solution
            should be reported. When omitted, the solver reports its internally
            selected integration points.

        Returns
        -------
        SimulationResult
            Solver-independent OPUS simulation result.

        Raises
        ------
        TypeError
            If the supplied model does not satisfy ``DynamicModel``.

        ValueError
            If the initial state, time span, or evaluation times are invalid.
        """

        self._validate_model(model)

        validated_initial_state = self._validate_initial_state(initial_state)
        validated_time_span = self._validate_time_span(time_span)
        validated_evaluation_times = self._validate_evaluation_times(
            evaluation_times=evaluation_times,
            time_span=validated_time_span,
        )

        solution = solve_ivp(
            fun=model.rhs,
            t_span=validated_time_span,
            y0=validated_initial_state,
            method=self.method,
            t_eval=validated_evaluation_times,
            rtol=self.relative_tolerance,
            atol=self.absolute_tolerance,
            max_step=self.max_step,
        )

        return SimulationResult(
            time=solution.t,
            states=solution.y,
            success=solution.success,
            message=solution.message,
            nfev=solution.nfev,
            njev=solution.njev,
            nlu=solution.nlu,
        )

    @staticmethod
    def _validate_model(model: DynamicModel) -> None:
        """Validate that the supplied object is a dynamic model."""

        if not isinstance(model, DynamicModel):
            raise TypeError(
                "model must implement the DynamicModel protocol, including "
                "an rhs(t, y) method."
            )

    @staticmethod
    def _validate_initial_state(
        initial_state: StateVector,
    ) -> StateVector:
        """Validate and copy the initial state vector."""

        state = np.asarray(initial_state, dtype=np.float64)

        if state.ndim != 1:
            raise ValueError(
                "initial_state must be one-dimensional; "
                f"received shape {state.shape}."
            )

        if state.size == 0:
            raise ValueError("initial_state cannot be empty.")

        if not np.all(np.isfinite(state)):
            raise ValueError(
                "initial_state contains non-finite values."
            )

        return state.copy()

    @staticmethod
    def _validate_time_span(
        time_span: TimeSpan,
    ) -> TimeSpan:
        """Validate the simulation time interval."""

        if len(time_span) != 2:
            raise ValueError(
                "time_span must contain exactly two values: "
                "(initial_time, final_time)."
            )

        initial_time = float(time_span[0])
        final_time = float(time_span[1])

        if not np.isfinite(initial_time) or not np.isfinite(final_time):
            raise ValueError(
                "time_span values must be finite."
            )

        if final_time <= initial_time:
            raise ValueError(
                "final_time must be greater than initial_time."
            )

        return initial_time, final_time

    @staticmethod
    def _validate_evaluation_times(
        *,
        evaluation_times: TimeVector | None,
        time_span: TimeSpan,
    ) -> TimeVector | None:
        """Validate and copy requested output times."""

        if evaluation_times is None:
            return None

        times = np.asarray(evaluation_times, dtype=np.float64)

        if times.ndim != 1:
            raise ValueError(
                "evaluation_times must be one-dimensional; "
                f"received shape {times.shape}."
            )

        if times.size == 0:
            raise ValueError("evaluation_times cannot be empty.")

        if not np.all(np.isfinite(times)):
            raise ValueError(
                "evaluation_times contains non-finite values."
            )

        if np.any(np.diff(times) <= 0.0):
            raise ValueError(
                "evaluation_times must be strictly increasing."
            )

        initial_time, final_time = time_span

        if times[0] < initial_time or times[-1] > final_time:
            raise ValueError(
                "All evaluation_times must lie within time_span."
            )

        return times.copy()