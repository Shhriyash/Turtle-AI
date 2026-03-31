"""Tests for Zusatzinfo classification result writing."""

import json

from KIPPSFlow.parsing.parseddata import Basisinformation, ProzessContent
from KIPPSFlow.zusatzinfo.classification_result import write_classification_results


def test_write_classification_results_handles_unicode_characters(
    mock_run_context, tmp_path
) -> None:
    """Classification JSON is written as UTF-8 and preserves Unicode text."""
    mock_run_context.run_out_dir = tmp_path / "run_out"
    mock_run_context.run_out_dir.mkdir(parents=True)

    prozess = ProzessContent(
        title="Test",
        prozess_nr="1.02.55.610",
        variant="",
        basisinfo=Basisinformation(
            gliederungsnummer="1.02.55.610",
            bezeichnung="Test",
            release="1.8",
            fachverantwortliche="",
            abteilung="",
        ),
        zusatzinformationen=[
            {"chunk_id": 0, "chunk": "A -> B \u2192 C", "is_table": False, "is_img": False}
        ],
    )

    classification_result = [
        {"chunk_id": 0, "category": "TRANSFER", "reason": "Unicode \u2192 kept"}
    ]

    write_classification_results(mock_run_context, prozess, classification_result)

    out_path = (
        mock_run_context.run_out_dir
        / "zusatzinfo_klassifizierung"
        / "1.02.55.610.json"
    )
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["TRANSFER"]["reason"] == "Unicode \u2192 kept"
    assert payload["TRANSFER"]["chunk"] == "A -> B \u2192 C"