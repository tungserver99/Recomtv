import json

from recsys_vnpt.batch_cli import main


def test_make_ready_cli_assigns_batch_start_without_cutoff_argument(tmp_path, capsys):
    batch = tmp_path / "slot-1"
    for relative in (
        "duration/part.csv",
        "mytv_vmp_content/content.json",
        "tags/TAG.json",
        "content_tags/CONTENT_TO_TAG.json",
        "month_mytv_info.parquet",
    ):
        path = batch / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)

    exit_code = main(["make-ready", "--batch-dir", str(batch)])

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["batch_started_at"].endswith("+00:00")
    assert "data_cutoff" not in result


def test_make_ready_cli_rejects_manual_cutoff_argument(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        main(["make-ready", "--batch-dir", str(tmp_path), "--data-cutoff", "2025-03-01T00:00:00+00:00"])
