"""
Simulation result objects used throughout OPUS.

This module defines solver-independent containers for simulation outputs.
Numerical backends such as SciPy should convert their native result objects
into these classes before returning results to users.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


TimeVector = NDArray[np.float64]
StateVector = NDArray[np.float64]
StateMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """
    Results from a dynamic process simulation.

    The state matrix follows the convention used by SciPy ODE solvers:

        states[i, j]

    is the value of state variable ``i`` at time point ``j``.

    Parameters
    ----------
    time
        One-dimensional array of simulation time points.

    states
        Two-dimensional state matrix with shape
        ``(number_of_states, number_of_time_points)``.

    success
        Whether the numerical solver completed successfully.

    message
        Human-readable termination message from the solver.

    nfev
        Number of right-hand-side function evaluations.

    njev
        Number of Jacobian evaluations, when reported by the solver.

    nlu
        Number of LU decompositions, when reported by the solver.
    """

    time: TimeVector
    states: StateMatrix
    success: bool
    message: str
    nfev: int
    njev: int | None = None
    nlu: int | None = None

    def __post_init__(self) -> None:
        """
        Validate and protect the simulation output arrays.
        """

        time = np.asarray(self.time, dtype=np.float64)
        states = np.asarray(self.states, dtype=np.float64)

        self._validate_arrays(time=time, states=states)
        self._validate_solver_statistics()

        # Copy the arrays so external mutations cannot affect the result.
        time = time.copy()
        states = states.copy()

        # Frozen dataclasses do not make NumPy arrays immutable by themselves.
        time.setflags(write=False)
        states.setflags(write=False)

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "states", states)

    @staticmethod
    def _validate_arrays(
        *,
        time: TimeVector,
        states: StateMatrix,
    ) -> None:
        """Validate time and state-array dimensions and values."""

        if time.ndim != 1:
            raise ValueError(
                "The time vector must be one-dimensional; "
                f"received shape {time.shape}."
            )

        if states.ndim != 2:
            raise ValueError(
                "The state matrix must be two-dimensional; "
                f"received shape {states.shape}."
            )

        if time.size == 0:
            raise ValueError("The time vector cannot be empty.")

        if states.shape[0] == 0:
            raise ValueError(
                "The state matrix must contain at least one state variable."
            )

        if states.shape[1] != time.size:
            raise ValueError(
                "The number of state-matrix columns must equal the number "
                f"of time points; received {states.shape[1]} state columns "
                f"and {time.size} time points."
            )

        if not np.all(np.isfinite(time)):
            raise ValueError("The time vector contains non-finite values.")

        if not np.all(np.isfinite(states)):
            raise ValueError("The state matrix contains non-finite values.")

        if np.any(np.diff(time) <= 0.0):
            raise ValueError(
                "Simulation time points must be strictly increasing."
            )

    def _validate_solver_statistics(self) -> None:
        """Validate solver diagnostic counts."""

        if self.nfev < 0:
            raise ValueError("nfev cannot be negative.")

        if self.njev is not None and self.njev < 0:
            raise ValueError("njev cannot be negative.")

        if self.nlu is not None and self.nlu < 0:
            raise ValueError("nlu cannot be negative.")

    @property
    def number_of_states(self) -> int:
        """Number of dynamic state variables."""

        return self.states.shape[0]

    @property
    def number_of_time_points(self) -> int:
        """Number of reported simulation time points."""

        return self.time.size

    @property
    def initial_state(self) -> StateVector:
        """
        Return a copy of the initial state vector.
        """

        return self.states[:, 0].copy()

    @property
    def final_state(self) -> StateVector:
        """
        Return a copy of the final state vector.
        """

        return self.states[:, -1].copy()

    @property
    def final_time(self) -> float:
        """Final reported simulation time."""

        return float(self.time[-1])

    def state_trajectory(self, index: int) -> StateVector:
        """
        Return a copy of one state's trajectory.

        Parameters
        ----------
        index
            Zero-based state-variable index.

        Returns
        -------
        StateVector
            Values of the selected state at every reported time point.
        """

        if not 0 <= index < self.number_of_states:
            raise IndexError(
                f"State index {index} is out of range for "
                f"{self.number_of_states} states."
            )

        return self.states[index, :].copy()