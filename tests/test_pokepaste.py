from vgc_team.teams.pokepaste import parse_pokepaste

PASTE = """Charizard @ Charizardite Y
Ability: Blaze
Level: 50
Tera Type: Fire
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Heat Wave
- Solar Beam
- Protect
- Weather Ball

Sneaky (Sneasler) (M) @ Focus Sash
Ability: Unburden
- Close Combat
- Dire Claw
- Fake Out
- Protect

Rotom-Wash
Ability: Levitate
- Hydro Pump
- Thunderbolt
- Protect
- Will-O-Wisp
"""


def test_parse_pokepaste_handles_varied_headers() -> None:
    team = parse_pokepaste(PASTE)
    assert len(team) == 3

    charizard = team[0]
    assert charizard.species == "Charizard"
    assert charizard.item == "Charizardite Y"
    assert charizard.ability == "Blaze"
    assert charizard.nature == "Timid"
    assert charizard.level == 50
    assert charizard.moves == ("Heat Wave", "Solar Beam", "Protect", "Weather Ball")

    sneasler = team[1]
    assert sneasler.species == "Sneasler"  # extracted from "(Sneasler)" nickname form
    assert sneasler.gender == "M"
    assert sneasler.item == "Focus Sash"

    rotom = team[2]
    assert rotom.species == "Rotom-Wash"
    assert rotom.item == ""  # no item
    assert len(rotom.moves) == 4


def test_parse_pokepaste_ignores_evs_and_blank_blocks() -> None:
    team = parse_pokepaste("\n\n" + PASTE + "\n\n")
    assert len(team) == 3
    # EVs line must not leak into any field
    for mon in team:
        assert "EV" not in mon.nature
        assert all("/" not in move for move in mon.moves)
