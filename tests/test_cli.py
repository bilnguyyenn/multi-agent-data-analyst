import pytest

from analyst.cli import main


def test_top_level_help_renders_all_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "trap-variance" in output
    assert "scorecard" in output
    assert "heldout-eval" in output
    assert "heldout-scorecard" in output
    assert "gen-heldout-data" in output


def test_generate_heldout_data_command(tmp_path, capsys):
    output = tmp_path / "heldout"

    result = main(["gen-heldout-data", "--out", str(output), "--seed", "123"])

    assert result == 0
    assert len(list(output.glob("*.csv"))) == 6
    assert (tmp_path / "heldout.metadata.json").exists()
    assert "6 held-out tables" in capsys.readouterr().out
