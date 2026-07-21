import numpy as np

from opus.core.interfaces import DynamicModel, StateVector


class DummyDynamicModel:
    def rhs(
        self,
        t: float,
        y: StateVector,
    ) -> StateVector:
        return -y


def test_compatible_model_satisfies_dynamic_model_protocol() -> None:
    model = DummyDynamicModel()

    assert isinstance(model, DynamicModel)


def test_dynamic_model_rhs_returns_expected_derivative() -> None:
    model = DummyDynamicModel()
    state = np.array([1.0, 2.0], dtype=np.float64)

    derivative = model.rhs(0.0, state)

    np.testing.assert_allclose(
        derivative,
        np.array([-1.0, -2.0], dtype=np.float64),
    )