"""
State objects for dynamic process models.
"""

from __future__ import annotations
from enum import IntEnum
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

class CSTRStateIndex(IntEnum):
    """
    Indices of the reactor state vector.

    The enum provides symbolic names for each element of the
    state vector used by SciPy ODE solvers.
    """

    CONCENTRATION = 0
    TEMPERATURE = 1

class ReactorState(BaseModel):
    """
    Dynamic state of an ideal CSTR.

    Attributes
    ----------
    concentration
        Reactant concentration [mol/m³]

    temperature
        Reactor temperature [K]
    """

    model_config = ConfigDict(frozen=True)

    concentration: float
    temperature: float

    @classmethod
    def from_vector(
        cls,
        y: NDArray[np.float64],
    ) -> "ReactorState":
        """
        Construct a ReactorState from a NumPy state vector.
        """

        if y.shape != (2,):
            raise ValueError(
                f"Expected state vector of shape (2,), got {y.shape}"
            )

        return cls(
            concentration=float(y[CSTRStateIndex.CONCENTRATION]),
            temperature=float(y[CSTRStateIndex.TEMPERATURE]),
        )

    def to_vector(self) -> NDArray[np.float64]:
        """
        Convert the state to a NumPy vector.
        """

        return np.asarray(
            [
                self.concentration,
                self.temperature,
            ],
            dtype=np.float64,
        )