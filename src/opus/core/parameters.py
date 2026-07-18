"""
parameters.py

Data models used throughout OPUS.

This module defines

- Reactor parameters
- Feed parameters
- Cooling parameters
- Kinetic parameters
- Top-level simulation parameters
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Reactor
# =============================================================================


class ReactorParameters(BaseModel):
    """Physical properties of the reactor."""

    model_config = ConfigDict(frozen=True)

    volume: Annotated[
        float,
        Field(gt=0, description="Reactor volume [m³]")
    ]

    density: Annotated[
        float,
        Field(gt=0, description="Fluid density [kg/m³]")
    ]

    heat_capacity: Annotated[
        float,
        Field(gt=0, description="Heat capacity [J/(kg·K)]")
    ]

    UA: Annotated[
        float,
        Field(gt=0, description="Overall heat transfer coefficient × area [W/K]")
    ]


# =============================================================================
# Feed
# =============================================================================


class FeedParameters(BaseModel):
    """Feed stream properties."""

    model_config = ConfigDict(frozen=True)

    flow_rate: Annotated[
        float,
        Field(gt=0, description="Volumetric flow rate [m³/s]")
    ]

    concentration: Annotated[
        float,
        Field(gt=0, description="Feed concentration [mol/m³]")
    ]

    temperature: Annotated[
        float,
        Field(gt=0, description="Feed temperature [K]")
    ]


# =============================================================================
# Cooling
# =============================================================================


class CoolingParameters(BaseModel):
    """Cooling jacket properties."""

    model_config = ConfigDict(frozen=True)

    coolant_temperature: Annotated[
        float,
        Field(gt=0, description="Coolant temperature [K]")
    ]


# =============================================================================
# Kinetics
# =============================================================================


class KineticParameters(BaseModel):
    """Reaction kinetic parameters."""

    model_config = ConfigDict(frozen=True)

    k0: Annotated[
        float,
        Field(gt=0, description="Pre-exponential factor [1/s]")
    ]

    activation_energy: Annotated[
        float,
        Field(gt=0, description="Activation energy [J/mol]")
    ]

    heat_of_reaction: Annotated[
        float,
        Field(
            description="Heat of reaction ΔH [J/mol] "
                        "(negative for exothermic reactions)."
        )
    ]

    gas_constant: Annotated[
        float,
        Field(gt=0, description="Universal gas constant [J/mol/K]")
    ]


# =============================================================================
# Simulation Parameters
# =============================================================================


class SimulationParameters(BaseModel):
    """
    Top-level simulation parameters.
    """

    model_config = ConfigDict(frozen=True)

    reactor: ReactorParameters

    feed: FeedParameters

    cooling: CoolingParameters

    kinetics: KineticParameters