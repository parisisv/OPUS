"""
Reproducible numerical simulation configuration.

Physical process parameters belong in ``SimulationParameters``. The classes
in this module describe how a numerical simulation should be executed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from opus.models.state import ReactorState


SolverMethod = Literal[
    "RK23",
    "RK45",
    "DOP853",
    "Radau",
    "BDF",
    "LSODA",
]

TimeVector = NDArray[np.float64]
TimeSpan = tuple[float, float]


class SolverConfiguration(BaseModel):
    """
    Numerical ODE solver settings.

    Parameters
    ----------
    method
        Integration method passed to the ODE solver.

    relative_tolerance
        Relative error tolerance.

    absolute_tolerance
        Absolute error tolerance applied to each state variable.

    max_step
        Maximum integration step. When omitted, no explicit upper bound is
        imposed beyond the numerical solver's own step-size selection.
    """

    model_config = ConfigDict(frozen=True)

    method: SolverMethod = "RK45"

    relative_tolerance: float = Field(
        default=1.0e-6,
        gt=0.0,
    )

    absolute_tolerance: float = Field(
        default=1.0e-9,
        gt=0.0,
    )

    max_step: float | None = Field(
        default=None,
        gt=0.0,
    )

    @property
    def resolved_max_step(self) -> float:
        """
        Return the maximum step in the form expected by ``ODESolver``.
        """

        if self.max_step is None:
            return float(np.inf)

        return self.max_step


class TimeConfiguration(BaseModel):
    """
    Simulation time interval and output grid.

    Parameters
    ----------
    initial_time
        Beginning of the simulation interval.

    final_time
        End of the simulation interval.

    number_of_evaluation_points
        Number of evenly spaced reporting points, including both endpoints.
        When omitted, the ODE solver returns its internally selected points.
    """

    model_config = ConfigDict(frozen=True)

    initial_time: float
    final_time: float

    number_of_evaluation_points: int | None = Field(
        default=None,
        ge=2,
    )

    @model_validator(mode="after")
    def validate_time_interval(self) -> TimeConfiguration:
        """Require finite times and a forward simulation interval."""

        if not np.isfinite(self.initial_time):
            raise ValueError("initial_time must be finite.")

        if not np.isfinite(self.final_time):
            raise ValueError("final_time must be finite.")

        if self.final_time <= self.initial_time:
            raise ValueError(
                "final_time must be greater than initial_time."
            )

        return self

    @property
    def time_span(self) -> TimeSpan:
        """Return the simulation interval as a two-element tuple."""

        return self.initial_time, self.final_time

    @property
    def duration(self) -> float:
        """Return the total simulation duration."""

        return self.final_time - self.initial_time

    @property
    def evaluation_times(self) -> TimeVector | None:
        """
        Generate the evenly spaced output grid.

        Returns ``None`` when no reporting grid has been requested.
        """

        if self.number_of_evaluation_points is None:
            return None

        times = np.linspace(
            self.initial_time,
            self.final_time,
            self.number_of_evaluation_points,
            dtype=np.float64,
        )

        times.setflags(write=False)
        return times


class SimulationConfiguration(BaseModel):
    """
    Complete reproducible numerical configuration for a reactor simulation.

    This class describes the numerical experiment, while
    ``SimulationParameters`` describes the physical process.
    """

    model_config = ConfigDict(frozen=True)

    time: TimeConfiguration
    solver: SolverConfiguration = Field(
        default_factory=SolverConfiguration
    )
    initial_state: ReactorState

    @property
    def time_span(self) -> TimeSpan:
        """Return the configured simulation time interval."""

        return self.time.time_span

    @property
    def evaluation_times(self) -> TimeVector | None:
        """Return the requested output grid."""

        return self.time.evaluation_times