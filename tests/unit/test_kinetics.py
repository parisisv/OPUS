"""
Unit tests for kinetics.py
"""

import numpy as np
import pytest
from opus.core.constants import R_DEFAULT
from opus.kinetics.kinetics import (
    arrhenius_rate_constant,
    first_order_rate,
    reaction_heat,
)

# =============================================================================
# Arrhenius Equation
# =============================================================================


def test_arrhenius_rate_constant_increases_with_temperature():
    """
    The Arrhenius rate constant should increase with temperature.
    """

    k0 = 7.2e10
    ea = 8.0e4

    k300 = arrhenius_rate_constant(300.0, k0, ea)
    k350 = arrhenius_rate_constant(350.0, k0, ea)

    assert k350 > k300


def test_arrhenius_matches_hand_calculation():
    """
    Compare against the analytical Arrhenius equation.
    """

    temperature = 325.0
    k0 = 7.2e10
    ea = 8.0e4

    expected = k0 * np.exp(-ea / (R_DEFAULT * temperature))

    actual = arrhenius_rate_constant(
        temperature,
        k0,
        ea,
    )

    assert actual == pytest.approx(expected)


def test_arrhenius_rate_constant_is_positive():
    """
    The Arrhenius rate constant should always be positive.
    """

    k = arrhenius_rate_constant(
        temperature=350.0,
        k0=1.0e10,
        activation_energy=50000.0,
    )

    assert k > 0.0


# =============================================================================
# First-order reaction rate
# =============================================================================


def test_zero_concentration_gives_zero_rate():
    """
    Zero reactant concentration should produce zero reaction rate.
    """

    rate = first_order_rate(
        concentration=0.0,
        temperature=350.0,
        k0=1.0e10,
        activation_energy=50000.0,
    )

    assert rate == pytest.approx(0.0)


def test_rate_is_linear_with_concentration():
    """
    First-order kinetics should be linear in concentration.
    """

    r1 = first_order_rate(
        concentration=1000.0,
        temperature=350.0,
        k0=1.0e10,
        activation_energy=50000.0,
    )

    r2 = first_order_rate(
        concentration=2000.0,
        temperature=350.0,
        k0=1.0e10,
        activation_energy=50000.0,
    )

    assert r2 == pytest.approx(2.0 * r1)


def test_rate_increases_with_temperature():
    """
    At fixed concentration the reaction rate should increase
    with increasing temperature.
    """

    concentration = 1500.0

    r300 = first_order_rate(
        concentration,
        300.0,
        1.0e10,
        50000.0,
    )

    r350 = first_order_rate(
        concentration,
        350.0,
        1.0e10,
        50000.0,
    )

    assert r350 > r300


# =============================================================================
# Heat generation
# =============================================================================


def test_exothermic_reaction_generates_positive_heat():
    """
    Exothermic reactions (ΔH < 0) should produce positive heat generation.
    """

    q = reaction_heat(
        reaction_rate=5.0,
        heat_of_reaction=-50000.0,
    )

    assert q > 0.0


def test_endothermic_reaction_generates_negative_heat():
    """
    Endothermic reactions (ΔH > 0) should consume heat.
    """

    q = reaction_heat(
        reaction_rate=5.0,
        heat_of_reaction=50000.0,
    )

    assert q < 0.0


def test_zero_rate_produces_zero_heat():
    """
    No reaction means no heat generation.
    """

    q = reaction_heat(
        reaction_rate=0.0,
        heat_of_reaction=-50000.0,
    )

    assert q == pytest.approx(0.0)


# =============================================================================
# Regression tests
# =============================================================================


@pytest.mark.parametrize(
    "temperature",
    [290.0, 310.0, 330.0, 350.0, 370.0],
)
def test_rate_constant_is_finite(temperature):
    """
    The Arrhenius equation should never return NaN or Inf
    over the expected operating range.
    """

    k = arrhenius_rate_constant(
        temperature,
        7.2e10,
        80000.0,
    )

    assert np.isfinite(k)


@pytest.mark.parametrize(
    "concentration",
    [0.0, 500.0, 1000.0, 2000.0],
)
def test_rate_is_finite(concentration):
    """
    Reaction rate should remain finite.
    """

    r = first_order_rate(
        concentration,
        350.0,
        7.2e10,
        80000.0,
    )

    assert np.isfinite(r)