"""
kinetics.py

Reaction kinetics library for OPUS.

This module contains kinetic models that are independent of any
particular reactor type.

Units
-----
Temperature          K
Concentration        mol/m³
Rate constant        1/s
Reaction rate        mol/(m³·s)
Activation energy    J/mol
Heat of reaction     J/mol
"""

from __future__ import annotations

import numpy as np
from opus.core.constants import R_DEFAULT



def arrhenius_rate_constant(
    temperature: float,
    k0: float,
    activation_energy: float,
    gas_constant: float = R_DEFAULT,
) -> float:
    """
    Compute the Arrhenius rate constant.

    Parameters
    ----------
    temperature
        Reactor temperature [K]

    k0
        Pre-exponential factor [1/s]

    activation_energy
        Activation energy [J/mol]

    gas_constant
        Universal gas constant [J/mol/K]

    Returns
    -------
    float
        Rate constant [1/s]
    """

    return k0 * np.exp(
        -activation_energy /
        (gas_constant * temperature)
    )


def first_order_rate(
    concentration: float,
    temperature: float,
    k0: float,
    activation_energy: float,
    gas_constant: float = R_DEFAULT,
) -> float:
    """
    Compute the first-order reaction rate.

    r = k C_A

    Parameters
    ----------
    concentration
        Reactant concentration [mol/m³]

    temperature
        Reactor temperature [K]

    Returns
    -------
    float
        Reaction rate [mol/(m³·s)]
    """

    k = arrhenius_rate_constant(
        temperature,
        k0,
        activation_energy,
        gas_constant,
    )

    return k * concentration


def reaction_heat(
    reaction_rate: float,
    heat_of_reaction: float,
) -> float:
    """
    Compute volumetric heat generation.

    q = -ΔH · r

    Parameters
    ----------
    reaction_rate
        Reaction rate [mol/(m³·s)]

    heat_of_reaction
        Heat of reaction ΔH [J/mol]

    Returns
    -------
    float
        Heat generation [W/m³]
    """

    return -heat_of_reaction * reaction_rate