from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_without_assets_has_actionable_setup_message(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_WORKSPACE", str(tmp_path))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Unattended luggage watch"
    assert any("uv run vision prepare" in message.value for message in app.info)
    assert app.button[0].disabled
