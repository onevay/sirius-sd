import shutil
import subprocess

import pytest

from sd import proxy as PX


def _make_video(path):
    exe = PX.ffmpeg_exe()
    subprocess.run([exe, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2", "-pix_fmt", "yuv420p", str(path)], check=True)


@pytest.mark.skipif(not (shutil.which("ffmpeg") or __import__("importlib").util.find_spec("imageio_ffmpeg")), reason="нет ffmpeg")
def test_proxy_is_created_once_and_invalidated_by_change(tmp_path):
    v = tmp_path / "clip.mp4"
    _make_video(v)
    out = PX.ensure_proxy(v, height=120, fps=10, root=tmp_path / "px")
    assert out.exists() and out.suffix == ".mp4" and out.stat().st_size > 0
    t = out.stat().st_mtime_ns
    assert PX.ensure_proxy(v, height=120, fps=10, root=tmp_path / "px") == out and out.stat().st_mtime_ns == t     # повторно не перекодируется
    assert not list((tmp_path / "px").glob("*.part*"))
    _make_video(tmp_path / "other.mp4")
    v.write_bytes((tmp_path / "other.mp4").read_bytes() + b"\0")                                                    # исходник изменился
    assert PX.proxy_path(v, 120, 10, root=tmp_path / "px") != out


def test_proxy_error_is_readable(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="ffmpeg не смог"):
        PX.ensure_proxy(bad, root=tmp_path / "px")
    assert not list((tmp_path / "px").glob("*.part*"))
