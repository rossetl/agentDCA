import numpy as np
import pytest

from agentdca.server import _top_contacts


def test_top_contacts_returns_ranked_upper_triangle():
    scores = np.array(
        [
            [0.0, 0.2, 0.9],
            [0.2, 0.0, 0.5],
            [0.9, 0.5, 0.0],
        ]
    )

    contacts = _top_contacts(scores, top_k=2, min_sequence_separation=1)

    assert [(item["position_i"], item["position_j"]) for item in contacts] == [
        (0, 2),
        (1, 2),
    ]
    assert contacts[0]["position_i_1based"] == 1
    assert contacts[0]["position_j_1based"] == 3


@pytest.mark.parametrize("top_k,separation", [(0, 1), (1, 0)])
def test_top_contacts_validates_bounds(top_k, separation):
    with pytest.raises(ValueError):
        _top_contacts(
            np.zeros((2, 2)),
            top_k=top_k,
            min_sequence_separation=separation,
        )
