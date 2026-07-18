"""
OPUS - Open Process Uncertainty Simulator

A Python package for Monte Carlo simulation of Continuous Stirred-Tank Reactors
(CSTR) with comprehensive support for kinetic modeling, dynamic simulation, and
uncertainty quantification.
"""

__version__ = "0.1.0"
__author__ = "Vassilis Parisis"

from opus.core.parameters import (
    SimulationParameters,
    ReactorParameters,
    FeedParameters,
    CoolingParameters,
    KineticParameters,
)
from opus.models.cstr import CSTR
from opus.models.state import ReactorState
from opus.solvers import ODESolver, ODESolution
from opus.kinetics.kinetics import (
    arrhenius_rate_constant,
    first_order_rate,
    reaction_heat,
)

__all__ = [
    "SimulationParameters",
    "ReactorParameters",
    "FeedParameters",
    "CoolingParameters",
    "KineticParameters",
    "CSTR",
    "ReactorState",
    "ODESolver",
    "ODESolution",
    "arrhenius_rate_constant",
    "first_order_rate",
    "reaction_heat",
]
