from cof_scout.agent.guards import GuardState, validate_action


def test_duplicate_incompatible_and_scaffold_guards():
    state = GuardState(evaluated={"x"}, scaffold_counts={"s": 2})
    assert validate_action(state, candidate_id="x", scaffold_id="q", decorated=False, compatible=True)[1] == "duplicate"
    assert validate_action(state, candidate_id="y", scaffold_id="q", decorated=False, compatible=False)[1] == "incompatible"
    assert validate_action(state, candidate_id="y", scaffold_id="s", decorated=False, compatible=True)[1] == "per_scaffold_limit"
