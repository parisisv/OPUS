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

        self.params = parameters

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
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

            [dCA/dt,
             dT/dt]
        """

        state = ReactorState.from_vector(y)

        CA = state.concentration
        T = state.temperature

        p = self.params

        # -----------------------------------------------------------------
        # Reaction kinetics
        # -----------------------------------------------------------------

        r = first_order_rate(
            concentration=CA,
            temperature=T,
            k0=p.kinetics.k0,
            activation_energy=p.kinetics.activation_energy,
            gas_constant=p.kinetics.gas_constant,
        )

        q_rxn = reaction_heat(
            reaction_rate=r,
            heat_of_reaction=p.kinetics.heat_of_reaction,
        )

        # -----------------------------------------------------------------
        # Mass balance
        # -----------------------------------------------------------------

        dCA_dt = (
            p.feed.flow_rate / p.reactor.volume
        ) * (
            p.feed.concentration - CA
        ) - r

        # -----------------------------------------------------------------
        # Energy balance
        # -----------------------------------------------------------------

        convection = (
            p.feed.flow_rate / p.reactor.volume
        ) * (
            p.feed.temperature - T
        )

        reaction = (
            q_rxn
            /
            (
                p.reactor.density
                * p.reactor.heat_capacity
            )
        )

        cooling = (
            p.reactor.UA
            /
            (
                p.reactor.density
                * p.reactor.heat_capacity
                * p.reactor.volume
            )
        ) * (
            T - p.cooling.coolant_temperature
        )

        dT_dt = (
            convection
            + reaction
            - cooling
        )

        return np.array(
            [
                dCA_dt,
                dT_dt,
            ],
            dtype=float,
        )