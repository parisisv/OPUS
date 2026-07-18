from opus.models.state import ReactorState


def test_round_trip_conversion():
    state = ReactorState(
        concentration=1.0,
        temperature=300.0
    )

    vector = state.to_vector()

    new_state = ReactorState.from_vector(vector)

    assert state == new_state