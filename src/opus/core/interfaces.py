"""
Common interfaces used throughout OPUS.

Protocols define the behavior expected from simulation components without
coupling the framework to specific implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


StateVector = NDArray[np.float64]


@runtime_checkable
class DynamicModel(Protocol):
    """
    Interface for dynamic models governed by ordinary differential equations.

    Any model that implements a compatible ``rhs`` method can be integrated
    by an OPUS ODE solver.

    The governing system has the form

        dy/dt = f(t, y)

    where ``y`` is the model state vector.
    """

    def rhs(self, t: float, y: StateVector,) -> StateVector:
        """
        Evaluate the time derivative of the model state.

        Parameters
        ----------
        t
            Current simulation time.

        y
            Current model state vector.

        Returns
        -------
        StateVector
            Time derivative of the state vector.
        """
        ...