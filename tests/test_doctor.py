"""`sd doctor`: проверка окружения читает только файлы и версии; здесь — подсчёт видео данных (не только .mp4) и разбор lock-файла."""
from sd import doctor


def test_doctor_counts_all_video_formats(tmp_path):
    for d, names in (("курение", ["a.mp4", "b.mkv", "notes.txt"]), ("лжекурение", ["c.wmv", "d.MP4"])):
        (tmp_path / "data" / d).mkdir(parents=True)
        for n in names:
            (tmp_path / "data" / d / n).write_bytes(b"x")
    r = doctor.check(tmp_path)
    assert r["data"] == {"курение": 2, "лжекурение": 2}                      # .mkv/.wmv и верхний регистр считаются, .txt — нет


def test_parse_lock_ignores_comments_and_extras(tmp_path):
    f = tmp_path / "req.txt"
    f.write_text("# комментарий\nnumpy==2.0.0  # версия\nuvicorn[standard]==0.30.0\n\nfoo>=1.0\n", encoding="utf-8")
    assert doctor.parse_lock(f) == {"numpy": "2.0.0", "uvicorn": "0.30.0"}
