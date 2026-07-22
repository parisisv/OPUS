import numpy as np
import pytest

from opus.analysis.cstr_performance import (
    SteadyStateTolerance,
    calculate_cstr_performance,
)
from opus.core.parameters import (
    CoolingParameters,
    FeedParameters,
    KineticParameters,
    ReactorParameters,
    SimulationParameters,
)
from opus.core.results import SimulationResult
from opus.models.cstr import CSTR


@pytest.fixture
def parameters() -> SimulationParameters:
    return SimulationParameters(
        reactor=ReactorParameters(
            volume=2.0,
            density=1000.0,
            heat_capacity=4184.0,
            UA=500.0,
        ),
        feed=FeedParameters(
            flow_rate=0.5,
            concentration=1000.0,
            temperature=300.0,
        ),
        cooling=CoolingParameters(
            coolant_temperature=295.0,
        ),
        kinetics=KineticParameters(
            k0=2.0,
            activation_energy=0.0,
            heat_of_reaction=-50_000.0,
            gas_constant=8.314462618,
        ),
    )


@pytest.fixture
def reactor(
    parameters: SimulationParameters,
) -> CSTR:
    return CSTR(parameters)


@pytest.fixture
def successful_result() -> SimulationResult:
    return SimulationResult(
        time=np.array([0.0, 2.0, 4.0]),
        states=np.array(
            [
                [900.0, 700.0, 500.0],
                [300.0, 320.0, 310.0],
            ]
        ),
        success=True,
        message="Success.",
        nfev=20,
    )


def test_calculates_hydraulic_metrics(
    reactor: CSTR,
    successful_result: SimulationResult,
) -> None:
    metrics = calculate_cstr_performance(
        model=reactor,
        result=successful_result,
    )

    assert metrics.residence_time == pytest.approx(4.0)
    assert metrics.simulated_residence_times == pytest.approx(1.0)


def test_calculates_final_conversion(
    reactor: CSTR,
    successful_result: SimulationResult,
) -> None:
    metrics = calculate_cstr_performance(
        model=reactor,
        result=successful_result,
    )

    assert metrics.feed_concentration == pytest.approx(1000.0)
    assert metrics.final_concentration == pytest.approx(500.0)
    assert metrics.final_conversion == pytest.approx(0.5)


def test_calculates_temperature_extrema(
    reactor: CSTR,
    successful_result: SimulationResult,
) -> None:
    metrics = calculate_cstr_performance(
        model=reactor,
        result=successful_result,
    )

    assert metrics.minimum_temperature == pytest.approx(300.0)
    assert metrics.maximum_temperature == pytest.approx(320.0)
    assert metrics.time_of_maximum_temperature == pytest.approx(2.0)


def test_calculates_final_reaction_rate(
    reactor: CSTR,
    successful_result: SimulationResult,
) -> None:
    metrics = calculate_cstr_performance(
        model=reactor,
        result=successful_result,
    )

    # activation_energy = 0, so k = k0 = 2.
    #assert metrics.final_reaction_rate == pytest.approx(1000.0)
    assert metrics.final_reaction_rate == pytest.approx(2*successful_result.final_state[0])
    
    # Rate multiplied by a reactor volume of 2.
    assert metrics.final_reactant_consumption_rate == pytest.approx(
        2000.0
    )


def test_reaction_heat_is_positive_for_exothermic_reaction(
    reactor: CSTR,
    successful_result: SimulationResult,
) -> None:
    metrics = calculate_cstr_performance(
        model=reactor,
        result=successful_result,
    )

    expected = 50_000.0 * 2000.0

    assert metrics.final_reaction_heat_rate == pytest.approx(expected)
    assert metrics.final_reaction_heat_rate > 0.0


def test_calculates_cooling_duty(
    reactor: CSTR,
    successful_result: SimulationResult,
) -> None:
    metrics = calculate_cstr_performance(
        model=reactor,
        result=successful_result,
    )

    expected_final_duty = 500.0 * (310.0 - 295.0)
    expected_peak_duty = 500.0 * (320.0 - 295.0)

    assert metrics.final_cooling_duty == pytest.approx(
        expected_final_duty
    )
    assert metrics.peak_cooling_duty == pytest.approx(
        expected_peak_duty
    )


def test_steady_state_diagnostics_match_model_rhs(
    reactor: CSTR,
    successful_result: SimulationResult,
) -> None:
    metrics = calculate_cstr_performance(
        model=reactor,
        result=successful_result,
        steady_state_tolerance=SteadyStateTolerance(
            concentration=1.0e-12,
            temperature=1.0e-12,
        ),
    )

    expected = reactor.rhs(
        successful_result.final_time,
        successful_result.final_state,
    )

    assert metrics.final_concentration_derivative == pytest.approx(
        expected[0]
    )
    assert metrics.final_temperature_derivative == pytest.approx(
        expected[1]
    )

    assert not metrics.is_steady_state


def test_steady_state_classification_uses_supplied_tolerances(
    reactor: CSTR,
    successful_result: SimulationResult,
) -> None:
    metrics = calculate_cstr_performance(
        model=reactor,
        result=successful_result,
        steady_state_tolerance=SteadyStateTolerance(
            concentration=1.0e10,
            temperature=1.0e10,
        ),
    )

    assert metrics.is_steady_state


@pytest.mark.parametrize(
    ("concentration", "temperature"),
    [
        (0.0, 1.0e-6),
        (-1.0, 1.0e-6),
        (1.0e-6, 0.0),
        (1.0e-6, -1.0),
        (np.inf, 1.0e-6),
        (1.0e-6, np.nan),
    ],
)
def test_rejects_invalid_steady_state_tolerances(
    concentration: float,
    temperature: float,
) -> None:
    with pytest.raises(ValueError):
        SteadyStateTolerance(
            concentration=concentration,
            temperature=temperature,
        )


def test_rejects_unsuccessful_simulation(
    reactor: CSTR,
) -> None:
    result = SimulationResult(
        time=np.array([0.0, 1.0]),
        states=np.array(
            [
                [1000.0, 900.0],
                [300.0, 301.0],
            ]
        ),
        success=False,
        message="Integration failed.",
        nfev=10,
    )

    with pytest.raises(ValueError, match="unsuccessful"):
        calculate_cstr_performance(
            model=reactor,
            result=result,
        )


def test_rejects_incompatible_number_of_states(
    reactor: CSTR,
) -> None:
    result = SimulationResult(
        time=np.array([0.0, 1.0]),
        states=np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ),
        success=True,
        message="Success.",
        nfev=10,
    )

    with pytest.raises(ValueError, match="exactly"):
        calculate_cstr_performance(
            model=reactor,
            result=result,
        )