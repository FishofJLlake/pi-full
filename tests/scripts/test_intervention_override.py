from opentau.scripts.get_advantage_and_percentiles import (
    apply_intervention_override,
)


def test_positive_intervention_overrides_effective_advantage_and_preserves_raw_input():
    raw_advantage = -0.375

    effective, source = apply_intervention_override(raw_advantage, intervention=1)

    assert raw_advantage == -0.375
    assert effective == 1.0
    assert source == "human_intervention_override"


def test_non_intervention_keeps_td_advantage():
    effective, source = apply_intervention_override(-0.375, intervention=0)

    assert effective == -0.375
    assert source == "td"
