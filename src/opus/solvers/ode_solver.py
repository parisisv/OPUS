"""
ODE Solver for dynamic reactor simulations.

This module provides a high-level interface for solving the differential
equations that describe reactor dynamics using SciPy's ODE integration methods.
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp, OdeSolution
from dataclasses import dataclass

from opus.models.cstr import CSTR
from opus.models.state import ReactorState


@dataclass
class ODESolution:
    """
    Solution from ODE integration.

    Attributes
    ----------
    t
        Time points [s]

    y
        State values at each time point.
        Shape: (n_states, n_timepoints)

    sol
        Continuous solution object from scipy (callable)

    status
        Integration status

    message
        Integration message
    """

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    sol: Optional[OdeSolution]
    status: int
    message: str


class ODESolver:
    """
    Solver for ordinary differential equations describing reactor dynamics.

    Parameters
    ----------
    model
        CSTR model with rhs() method

    method
        Integration method. Default is 'RK45' (4th/5th order Runge-Kutta).
        Other options: 'RK23', 'BDF', 'DOP853', 'LSODA', 'Radau'

    dense_output
        If True, returns a continuous solution object
        Default is True

    events
        Event functions to detect during integration
        Default is None

    Raises
    ------
    ValueError
        If method is not recognized
    """

    def __init__(
        self,
        model: CSTR,
        method: str = "RK45",
        dense_output: bool = True,
        events: Optional[Callable] = None,
    ):
        """Initialize the ODE solver."""

        valid_methods = {"RK45", "RK23", "BDF", "DOP853", "LSODA", "Radau"}
        if method not in valid_methods:
            raise ValueError(
                f"Method '{method}' not recognized. "
                f"Valid methods: {valid_methods}"
            )

        self.model = model
        self.method = method
        self.dense_output = dense_output
        self.events = events

    def solve(
        self,
        initial_state: ReactorState,
        t_span: tuple[float, float],
        t_eval: Optional[NDArray[np.float64]] = None,
        max_step: Optional[float] = None,
        rtol: float = 1e-3,
        atol: float = 1e-6,
    ) -> ODESolution:
        """
        Solve the ODE system.

        Parameters
        ----------
        initial_state
            Initial reactor state

        t_span
            Integration time interval (t0, tf) [s]

        t_eval
            Time points at which to return the solution [s].
            If None, the solver automatically selects points.

        max_step
            Maximum allowed step size [s].
            Default is None (no limit)

        rtol
            Relative tolerance for the solver.
            Default is 1e-3

        atol
            Absolute tolerance for the solver.
            Default is 1e-6

        Returns
        -------
        ODESolution
            Solution object containing time points, states, and metadata

        Raises
        ------
        RuntimeError
            If the integration fails
        """

        y0 = initial_state.to_vector()

        solution = solve_ivp(
            fun=self.model.rhs,
            t_span=t_span,
            y0=y0,
            method=self.method,
            t_eval=t_eval,
            dense_output=self.dense_output,
            events=self.events,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
        )

        if not solution.success:
            raise RuntimeError(
                f"ODE integration failed: {solution.message}"
            )

        return ODESolution(
            t=solution.t,
            y=solution.y,
            sol=solution.sol if self.dense_output else None,
            status=solution.status,
            message=solution.message,
        )

    def solve_to_events(
        self,
        initial_state: ReactorState,
        t_span: tuple[float, float],
        events: Callable,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_step: Optional[float] = None,
    ) -> ODESolution:
        """
        Solve ODE until events are triggered.

        Parameters
        ----------
        initial_state
            Initial reactor state

        t_span
            Integration time interval (t0, tf) [s]

        events
            Event function(s) to detect

        rtol
            Relative tolerance for the solver.
            Default is 1e-3

        atol
            Absolute tolerance for the solver.
            Default is 1e-6

        max_step
            Maximum allowed step size [s].
            Default is None (no limit)

        Returns
        -------
        ODESolution
            Solution object up to the first detected event

        Raises
        ------
        RuntimeError
            If the integration fails
        """

        if events is None:
            raise ValueError("events must be provided for solve_to_events")

        events.terminal = True
        events.direction = -1

        y0 = initial_state.to_vector()

        solution = solve_ivp(
            fun=self.model.rhs,
            t_span=t_span,
            y0=y0,
            method=self.method,
            dense_output=self.dense_output,
            events=events,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
        )

        if not solution.success:
            raise RuntimeError(
                f"ODE integration failed: {solution.message}"
            )

        return ODESolution(
            t=solution.t,
            y=solution.y,
            sol=solution.sol if self.dense_output else None,
            status=solution.status,
            message=solution.message,
        )

    def evaluate_solution(
        self,
        solution: ODESolution,
        t_new: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Evaluate the solution at new time points.

        Parameters
        ----------
        solution
            Solution object from solve()

        t_new
            New time points at which to evaluate [s]

        Returns
        -------
        ndarray
            State values at new time points.
            Shape: (n_states, len(t_new))

        Raises
        ------
        ValueError
            If solution doesn't have dense output
        """

        if solution.sol is None:
            raise ValueError(
                "Solution does not have dense output. "
                "Set dense_output=True when creating solver."
            )

        return solution.sol(t_new).y

    def get_final_state(self, solution: ODESolution) -> ReactorState:
        """
        Extract the final reactor state from the solution.

        Parameters
        ----------
        solution
            Solution object from solve()

        Returns
        -------
        ReactorState
            Final reactor state
        """

        y_final = solution.y[:, -1]
        return ReactorState.from_vector(y_final)

    def get_state_at_time(
        self,
        solution: ODESolution,
        t: float,
    ) -> ReactorState:
        """
        Get the reactor state at a specific time.

        Parameters
        ----------
        solution
            Solution object from solve()

        t
            Time point [s]

        Returns
        -------
        ReactorState
            Reactor state at time t

        Raises
        ------
        ValueError
            If solution doesn't have dense output or t is out of range
        """

        if solution.sol is None:
            raise ValueError(
                "Solution does not have dense output. "
                "Set dense_output=True when creating solver."
            )

        if not (solution.t[0] <= t <= solution.t[-1]):
            raise ValueError(
                f"Time {t} is outside solution range [{solution.t[0]}, "
                f"{solution.t[-1]}]"
            )

        y = solution.sol(t).y.flatten()
        return ReactorState.from_vector(y)
