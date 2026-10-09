"""Раздача видео браузеру: вне Streamlit маленькое вшивается, большое — понятное предупреждение."""
from sd.ui import media as M


def _file(tmp_path, name, mb):
    p = tmp_path / name
    p.write_bytes(b"\0" * int(mb * 1e6))
    return p


def test_embed_outside_streamlit_and_warn_for_huge(tmp_path, monkeypatch):
    src, w = M.media_src(_file(tmp_path, "s.mp4", 0.5))
    assert src.startswith("data:video/mp4;base64,") and w is None
    monkeypatch.setattr(M, "EMBED_MAX_MB", 1)
    src, w = M.media_src(_file(tmp_path, "b.mp4", 2))
    assert src is None and "Streamlit" in w


def test_missing_or_empty_file_warns(tmp_path):
    assert M.media_src(tmp_path / "нет.mp4")[0] is None
    e = tmp_path / "e.mp4"
    e.write_bytes(b"")
    assert M.media_src(e)[0] is None and "пуст" in M.media_src(e)[1]


def test_webm_mime(tmp_path):
    p = tmp_path / "a.webm"
    p.write_bytes(b"1234")
    assert M.media_src(p)[0].startswith("data:video/webm")
