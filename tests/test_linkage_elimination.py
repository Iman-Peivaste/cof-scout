from cof_scout.baselines.linkage_elimination import LinkageEliminationState


def test_corrected_elimination_rule_and_fixed_horizon():
    state = LinkageEliminationState(("a", "b", "c"), k=3)
    for _ in range(3):
        state.observe("a", hit=False)
    assert not state.eligible("a")
    state.observe("b", hit=True)
    for _ in range(10):
        state.observe("b", hit=False)
    assert state.eligible("b")
    for _ in range(3):
        state.observe("c", hit=False)
    assert len(state.active) == 1
    for _ in range(50 - 17):
        state.observe(next(iter(state.active)), hit=False)
    assert len(state.active) == 1
