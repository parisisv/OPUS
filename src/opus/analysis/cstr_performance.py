"""
Engineering performance calculations for CSTR simulations.

The functions in this module interpret completed simulations. They do not
participate in numerical integration and do not modify the underlying model
or result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opus.core.results import SimulationResult
from opus.models.cstr import CSTR
from opus.models.state import CSTRStateIndex


@dataclass(frozen=True, slots=True)
class SteadyStateTolerance:
    """
    Absolute derivative tolerances used for steady-state classification.

    Parameters
    ----------
    concentration
        Maximum absolute concentration derivative considered steady.

    temperature
        Maximum absolute temperature derivative considered steady.
    """

    concentration: float = 1.0e-6
    temperature: float = 1.0e-6

    def __post_init__(self) -> None:
        if not np.isfinite(self.concentration):
            raise ValueError(
                "The concentration tolerance must be finite."
            )

        if not np.isfinite(self.temperature):
            raise ValueError(
                "The temperature tolerance must be finite."
            )

        if self.concentration <= 0.0:
            raise ValueError(
                "The concentration tolerance must be greater than zero."
            )

        if self.temperature <= 0.0:
            raise ValueError(
                "The temperature tolerance must be greater than zero."
            )


@dataclass(frozen=True, slots=True)
class CSTRPerformanceMetrics:
    """
    Engineering metrics derived from a completed CSTR simulation.

    Concentration, temperature, time, rate, and duty units follow the unit
    system used by the supplied CSTR parameters.

    Positive reaction heat rate denotes heat released into the reactor.
    Positive cooling duty denotes heat removed from the reactor.
    """

    residence_time: float
    simulated_residence_times: float

    initial_concentration: float
    final_concentration: float
    feed_concentration: float
    final_conversion: float

    initial_temperature: float
    final_temperature: float
    minimum_temperature: float
    maximum_temperature: float
    time_of_maximum_temperature: float

    final_reaction_rate: float
    final_reactant_consumption_rate: float

    final_reaction_heat_rate: float
    maximum_reaction_heat_rate: float

    final_cooling_duty: float
    peak_cooling_duty: float

    final_concentration_derivative: float
    final_temperature_derivative: float

    concentration_balance_residual: float
    energy_balance_residual: float

    is_steady_state: bool


def calculate_cstr_performance(
    *,
    model: CSTR,
    result: SimulationResult,
    steady_state_tolerance: SteadyStateTolerance | None = None,
) -> CSTRPerformanceMetrics:
    """
    Calculate engineering metrics from a completed CSTR simulation.

    Parameters
    ----------
    model
        CSTR model used to generate the simulation.

    result
        Completed simulation result.

    steady_state_tolerance
        Optional derivative tolerances used to classify the final state.

    Returns
    -------
    CSTRPerformanceMetrics
        Calculated hydraulic, conversion, thermal, duty, and steady-state
        metrics.

    Raises
    ------
    ValueError
        If the simulation result is unsuccessful or incompatible with a CSTR.
    """

    tolerance = (
        steady_state_tolerance
        if steady_state_tolerance is not None
        else SteadyStateTolerance()
    )

    _validate_result(result)

    parameters = model.parameters

    reactor_parameters = parameters.reactor
    feed_parameters = parameters.feed
    cooling_parameters = parameters.cooling
    kinetic_parameters = parameters.kinetics

    concentration = result.states[
        CSTRStateIndex.CONCENTRATION
    ]
    temperature = result.states[
        CSTRStateIndex.TEMPERATURE
    ]

    residence_time = (
        reactor_parameters.volume
        / feed_parameters.flow_rate
    )

    simulation_duration = result.final_time - float(result.time[0])

    simulated_residence_times = (
        simulation_duration / residence_time
    )

    feed_concentration = feed_parameters.concentration
    final_concentration = float(concentration[-1])

    final_conversion = _calculate_conversion(
        feed_concentration=feed_concentration,
        outlet_concentration=final_concentration,
    )

    rate_constant = kinetic_parameters.k0 * np.exp(
        -kinetic_parameters.activation_energy
        / (
            kinetic_parameters.gas_constant
            * temperature
        )
    )

    reaction_rate = rate_constant * concentration

    reactant_consumption_rate = (
        reaction_rate * reactor_parameters.volume
    )

    reaction_heat_rate = (
        -kinetic_parameters.heat_of_reaction
        * reactant_consumption_rate
    )

    cooling_duty = (
        reactor_parameters.UA
        * (
            temperature
            - cooling_parameters.coolant_temperature
        )
    )

    maximum_temperature_index = int(np.argmax(temperature))

    final_derivative = model.rhs(
        result.final_time,
        result.final_state,
    )

    final_concentration_derivative = float(
        final_derivative[CSTRStateIndex.CONCENTRATION]
    )
    final_temperature_derivative = float(
        final_derivative[CSTRStateIndex.TEMPERATURE]
    )

    concentration_balance_residual = (
        final_concentration_derivative
    )
    energy_balance_residual = final_temperature_derivative

    is_steady_state = (
        abs(final_concentration_derivative)
        <= tolerance.concentration
        and abs(final_temperature_derivative)
        <= tolerance.temperature
    )

    return CSTRPerformanceMetrics(
        residence_time=float(residence_time),
        simulated_residence_times=float(simulated_residence_times),

        initial_concentration=float(concentration[0]),
        final_concentration=final_concentration,
        feed_concentration=float(feed_concentration),
        final_conversion=float(final_conversion),

        initial_temperature=float(temperature[0]),
        final_temperature=float(temperature[-1]),
        minimum_temperature=float(np.min(temperature)),
        maximum_temperature=float(np.max(temperature)),
        time_of_maximum_temperature=float(
            result.time[maximum_temperature_index]
        ),

        final_reaction_rate=float(reaction_rate[-1]),
        final_reactant_consumption_rate=float(
            reactant_consumption_rate[-1]
        ),

        final_reaction_heat_rate=float(reaction_heat_rate[-1]),
        maximum_reaction_heat_rate=float(
            np.max(reaction_heat_rate)
        ),

        final_cooling_duty=float(cooling_duty[-1]),
        peak_cooling_duty=float(np.max(cooling_duty)),

        final_concentration_derivative=(
            final_concentration_derivative
        ),
        final_temperature_derivative=(
            final_temperature_derivative
        ),

        concentration_balance_residual=(
            concentration_balance_residual
        ),
        energy_balance_residual=energy_balance_residual,

        is_steady_state=is_steady_state,
    )


def _calculate_conversion(
    *,
    feed_concentration: float,
    outlet_concentration: float,
) -> float:
    """Calculate instantaneous outlet conversion relative to the feed."""

    if feed_concentration <= 0.0:
        raise ValueError(
            "Feed concentration must be greater than zero to calculate "
            "conversion."
        )

    return (
        feed_concentration - outlet_concentration
    ) / feed_concentration


def _validate_result(result: SimulationResult) -> None:
    """Validate that a result can be interpreted as a CSTR simulation."""

    if not result.success:
        raise ValueError(
            "Engineering metrics cannot be calculated from an "
            "unsuccessful simulation."
        )

    expected_number_of_states = len(CSTRStateIndex)

    if result.number_of_states != expected_number_of_states:
        raise ValueError(
            "The CSTR performance calculator requires exactly "
            f"{expected_number_of_states} states; received "
            f"{result.number_of_states}."
        )