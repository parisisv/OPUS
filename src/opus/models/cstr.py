"""
Continuous Stirred-Tank Reactor (CSTR) model.

This module implements the dynamic mass and energy balances for an
ideal, perfectly mixed continuous stirred-tank reactor.
"""

from __future__ import annotations

import numpy as np

from opus.kinetics.kinetics import (
    first_order_rate,
    reaction_heat,
)
from opus.core.parameters import SimulationParameters
from opus.models.state import ReactorState
from opus.core.interfaces import StateVector

class CSTR:
    """
    Ideal Continuous Stirred-Tank Reactor.

    Parameters
    ----------
    parameters
        Simulation parameters describing reactor geometry,
        feed conditions, cooling system, and reaction kinetics.
    """

    def __init__(self, parameters: SimulationParameters):

        self.parameters = parameters

    def rhs(self, t: float, y: StateVector) -> StateVector:
        """
        Compute the time derivatives of the reactor state.

        Parameters
        ----------
        t
            Time [s].

            Present for compatibility with scipy.integrate.solve_ivp.

        y
            State vector

            y[0] = Reactant concentration [mol/m³]

            y[1] = Reactor temperature [K]

        Returns
        -------
        ndarray
            Time derivatives of the state vector
            [dCA/dt, dT/dt]
        """

        state = ReactorState.from_vector(y)

        CA = state.concentration
        T = state.temperature

        # p = self.params

        # -----------------------------------------------------------------
        # Reaction kinetics
        # -----------------------------------------------------------------

        r = first_order_rate(concentration=CA,
            temperature=T, k0=self.parameters.kinetics.k0, 
            activation_energy=self.parameters.kinetics.activation_energy,
            gas_constant=self.parameters.kinetics.gas_constant,
        )

        q_rxn = reaction_heat(
            reaction_rate=r,
            heat_of_reaction=self.parameters.kinetics.heat_of_reaction,
        )

        # -----------------------------------------------------------------
        # Mass balance
        # -----------------------------------------------------------------

        dCA_dt = (self.parameters.feed.flow_rate / self.parameters.reactor.volume) * (self.parameters.feed.concentration - CA) - r

        # -----------------------------------------------------------------
        # Energy balance
        # -----------------------------------------------------------------

        convection = (self.parameters.feed.flow_rate / self.parameters.reactor.volume) * (self.parameters.feed.temperature - T)

        reaction = (q_rxn/(self.parameters.reactor.density * self.parameters.reactor.heat_capacity))

        cooling = (self.parameters.reactor.UA / (self.parameters.reactor.density * self.parameters.reactor.heat_capacity * self.parameters.reactor.volume)) * \
                  ( T - self.parameters.cooling.coolant_temperature)

        dT_dt = (convection + reaction - cooling)

        return np.array(
            [dCA_dt, dT_dt],
            dtype=float,
        )