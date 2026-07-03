from vgc_team.teams.showdown import extract_open_sheet_teams, parse_packed_showteam


def test_parse_packed_showteam_extracts_core_fields() -> None:
    packed = (
        "Blaziken||FocusSash|SpeedBoost|HeatWave,AuraSphere,Coaching,Protect|Timid||F|||50|]"
        "Metagross||Metagrossite|ClearBody|IronHead,PsychicFangs,IcePunch,Protect|Jolly|||||50|"
    )

    team = parse_packed_showteam(packed)

    assert len(team) == 2
    assert team[0].species == "Blaziken"
    assert team[0].item == "FocusSash"
    assert team[0].ability == "SpeedBoost"
    assert team[0].moves == ("HeatWave", "AuraSphere", "Coaching", "Protect")
    assert team[0].nature == "Timid"
    assert team[0].level == 50


def test_extract_open_sheet_teams_attaches_winner() -> None:
    log = "\n".join(
        [
            "|player|p1|Alice|",
            "|player|p2|Bob|",
            "|showteam|p1|Raichu||FocusSash|LightningRod|FakeOut,VoltSwitch|Timid|||||50|",
            "|showteam|p2|Mawile||Mawilite|Intimidate|PlayRough,Protect|Adamant|||||50|",
            "|win|Alice",
        ]
    )

    teams = extract_open_sheet_teams(
        battle_id="battle-1",
        timestamp=123,
        format_id="gen9championsvgc2026regma",
        log=log,
    )

    assert len(teams) == 2
    assert teams[0].player == "Alice"
    assert teams[0].won is True
    assert teams[1].player == "Bob"
    assert teams[1].won is False
