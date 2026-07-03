import json

import pytest

from vgc_team.app.service import (
    DEFAULT_CHECKPOINT,
    DEFAULT_REFERENCE,
    MULTI_REG_DIR,
    AppState,
)

_have_artifacts = (
    DEFAULT_CHECKPOINT.exists()
    and DEFAULT_REFERENCE.exists()
    and any(MULTI_REG_DIR.glob("reg*.json"))
)
requires_artifacts = pytest.mark.skipif(
    not _have_artifacts, reason="checkpoint / reference / cached regs not present"
)


@requires_artifacts
def test_service_forecast_shape_and_json() -> None:
    state = AppState()
    state.refresh(source="cached")
    out = state.forecast(lam=0.3)

    assert out["n_teams"] > 0 and out["n_weeks"] >= 1
    assert len(out["cores"]) == state.reference.n_clusters
    core = out["cores"][0]
    assert {"current", "predicted", "delta", "representative", "label"} <= core.keys()
    assert abs((core["predicted"] - core["current"]) - core["delta"]) < 1e-9
    # coarse archetype families form a valid distribution
    assert out["archetypes"]
    assert abs(sum(a["current"] for a in out["archetypes"]) - 1.0) < 1e-6
    assert len(out["top_cores"]) <= 5 and len(out["top_archetypes"]) <= 5
    # fully JSON-serializable (no numpy types leak through)
    json.dumps(out)


_PASTE = """Incineroar @ Safety Goggles
Ability: Intimidate
- Fake Out
- Flare Blitz
- Parting Shot
- Knock Off

Amoonguss @ Sitrus Berry
Ability: Regenerator
- Spore
- Rage Powder
- Pollen Puff
- Protect
"""


@requires_artifacts
def test_rate_team_scores_and_aggregate() -> None:
    state = AppState()
    state.refresh(source="cached")
    out = state.rate_team(_PASTE, lam=0.3)
    if not out["ok"] and "matchup model not loaded" in out.get("error", ""):
        import pytest as _pytest
        _pytest.skip("matchup model not present")
    assert out["ok"]
    assert out["archetypes"] and all(0.0 <= a["winrate"] <= 1.0 for a in out["archetypes"])
    agg = out["aggregate"]
    assert 0.0 <= agg["current_by_cluster"] <= 1.0
    assert 0.0 <= agg["predicted_by_cluster"] <= 1.0
    json.dumps(out)


_PARTIAL = """Incineroar @ Safety Goggles
Ability: Intimidate
- Fake Out
- Flare Blitz
- Parting Shot

Garchomp
Ability: Rough Skin
- Earthquake
- Dragon Claw
- Protect
- Rock Slide
"""


@requires_artifacts
def test_recommend_completion() -> None:
    state = AppState()
    state.refresh(source="cached")
    out = state.recommend_completion(_PARTIAL, lam=0.3)
    if not out["ok"] and "matchup model not loaded" in out.get("error", ""):
        import pytest as _pytest
        _pytest.skip("matchup model not present")
    assert out["ok"]
    assert out["n_pokemon"] == 2 and not out["complete"]
    assert len(out["worst_archetypes"]) >= 1
    slots = {r["slot"] for r in out["recommendations"]}
    # a 2-mon team is missing Pokémon; Incineroar is short a move
    assert any("Pokémon" in s for s in slots)
    for rec in out["recommendations"]:
        assert rec["options"]
        for o in rec["options"]:
            assert 0.0 <= o["winrate_vs_worst"] <= 1.0
    json.dumps(out)


@requires_artifacts
def test_lambda_zero_predicts_current_week() -> None:
    state = AppState()
    state.refresh(source="cached")
    out = state.forecast(lam=0.0)
    # glide λ=0 => next = last week => zero predicted change everywhere
    assert max(abs(c["delta"]) for c in out["cores"]) < 1e-9
    assert max(abs(a["delta"]) for a in out["archetypes"]) < 1e-9
