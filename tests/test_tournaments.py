from vgc_team.data.tournaments import (
    TournamentInfo,
    load_tournament_teams,
    save_teams_json,
    standing_to_team,
)

INFO = TournamentInfo(
    id="abc123", name="Test Cup (Reg M-A)", date="2026-05-01T12:00:00.000Z",
    format="M-A", players=64,
)

STANDING = {
    "name": "alice",
    "player": "alice",
    "placing": 1,
    "decklist": [
        {
            "id": "ninetales-alola",
            "name": "Alolan Ninetales",
            "item": "Light Orb",
            "ability": "Snow Warning",
            "attacks": ["Blizzard", "Moonblast", "Aurora Veil", "Protect"],
            "nature": "Timid",
            "tera": None,
        },
        {
            "id": "incineroar",
            "name": "Incineroar",
            "item": "Safety Goggles",
            "ability": "Intimidate",
            "attacks": ["Fake Out", "Flare Blitz", "Parting Shot", "Knock Off"],
            "nature": "Careful",
            "tera": None,
        },
    ],
}


def test_standing_to_team_uses_canonical_id() -> None:
    team = standing_to_team(STANDING, INFO)
    assert team is not None
    # the canonical `id` field is used as species (matches pokedex keys), not display name
    assert team.pokemon[0].species == "ninetales-alola"
    assert team.pokemon[0].ability == "Snow Warning"
    assert team.pokemon[0].moves == ("Blizzard", "Moonblast", "Aurora Veil", "Protect")
    assert team.timestamp == INFO.timestamp
    assert team.format_id == "gen9championsvgc2026regma"


def test_standing_without_decklist_is_skipped() -> None:
    assert standing_to_team({"name": "bob", "decklist": None}, INFO) is None


def test_save_load_round_trip(tmp_path) -> None:
    team = standing_to_team(STANDING, INFO)
    path = tmp_path / "teams.json"
    save_teams_json([team], [8.0], path)
    teams, weights = load_tournament_teams(path)
    assert len(teams) == 1
    assert weights.tolist() == [8.0]
    assert teams[0].pokemon[0].species == "ninetales-alola"
    assert teams[0].timestamp == INFO.timestamp
