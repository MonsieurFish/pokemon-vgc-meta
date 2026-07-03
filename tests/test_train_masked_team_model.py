from scripts.train_masked_team_model import _relative_loss_improvement


def test_relative_loss_improvement() -> None:
    assert _relative_loss_improvement(10.0, 9.0) == 0.1
    assert _relative_loss_improvement(10.0, 10.1) < 0
    assert _relative_loss_improvement(0.0, 0.0) == 0.0
