"""Раздача видео браузеру: маленькое вшивается, большое публикуется файлом; каталог не растёт."""
import os

from sd.ui import media as M


def _file(tmp_path, name, mb):
    p = tmp_path / name
    p.write_bytes(b"\0" * int(mb * 1e6))
    return p


def test_small_embedded_large_published_and_warn_without_static(tmp_path):
    small, big, huge = _file(tmp_path, "s.mp4", 0.5), _file(tmp_path, "b.mp4", 8), _file(tmp_path, "h.mp4", 80)
    d = tmp_path / "static"
    src, w = M.media_src(small, static=True, static_dir=d)
    assert src.startswith("data:video/mp4;base64,") and w is None
    src, w = M.media_src(big, static=True, static_dir=d)
    assert src.startswith(M.URL_PREFIX) and w is None and len(list(d.iterdir())) == 1
    src, w = M.media_src(big, static=False, static_dir=d)
    assert src.startswith("data:") and w is None                     # без статической раздачи до лимита вшиваем
    src, w = M.media_src(huge, static=False, static_dir=d)
    assert src is None and "sd ui" in w                              # слишком большое и раздачи нет — понятное сообщение
    src, w = M.media_src(huge, static=True, static_dir=d)
    assert src.startswith(M.URL_PREFIX)


def test_publish_is_idempotent_updates_on_change_and_prunes(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "KEEP_FILES", 3)
    d = tmp_path / "st"
    p = _file(tmp_path, "a.mp4", 1)
    u1 = M.publish(p, d)
    assert M.publish(p, d).split("?")[0] == u1.split("?")[0] and len(list(d.iterdir())) == 1       # повтор не плодит файлы
    p.write_bytes(b"\1" * 2_000_000)
    os.utime(p, None)
    u2 = M.publish(p, d)
    assert u2.split("?")[0] != u1.split("?")[0]                                                       # изменённый файл — новый URL
    for i in range(5):
        M.publish(_file(tmp_path, f"x{i}.mp4", 1.1 + i / 10), d)
    assert len(list(d.iterdir())) <= 3 and not list(d.glob("*.part"))
