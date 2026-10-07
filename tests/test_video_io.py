"""iter_frames: выбор кадров по номерам (сетка кадров позы не совпадает с шагом от начала произвольного окна)."""
import cv2
import numpy as np
import pytest

from sd.video_io import iter_frames


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("vid") / "t.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    if not w.isOpened():
        pytest.skip("нет кодека MJPG в этой сборке OpenCV")
    for k in range(90):
        w.write(np.full((48, 64, 3), 2 * k, np.uint8))   # яркость кадра = 2·номер: так проверяем, что отдан именно тот кадр
    w.release()
    return p


def test_stride_is_counted_from_window_start_not_from_zero(video):
    aligned = [f.idx for f in iter_frames(video, 0.0, None, 3)]
    assert aligned[:4] == [0, 3, 6, 9]
    shifted = [f.idx for f in iter_frames(video, 0.7, 2.0, 3)]                    # окно начинается на кадре 21 -> 21, 24, ...
    assert shifted[0] == 21 and all(i % 3 == 0 for i in shifted)
    odd = [f.idx for f in iter_frames(video, 0.5, 1.5, 3)]                        # начало на кадре 15: 15 % 3 == 0, но у позы сетка могла быть 1, 4, 7, …
    assert odd[0] == 15


def test_only_idx_returns_exactly_the_pose_frames_for_any_window_start(video):
    pose_grid = set(range(1, 90, 3))                                              # запуск начался с кадра 1: сетка 1, 4, 7, …
    got = [f.idx for f in iter_frames(video, 0.7, 2.0, only_idx=pose_grid)]
    assert got and all(i in pose_grid for i in got)
    assert got[0] == 22 and got[-1] <= 61                                         # первый кадр сетки не раньше начала окна (кадр 21), последний не позже конца (2.0 с ≈ кадр 60)
    assert [f.idx for f in iter_frames(video, 0.7, 2.0, 3)] != got                # старый способ (stride от начала окна) даёт другие кадры и не пересекается с сеткой
    assert not set(f.idx for f in iter_frames(video, 0.7, 2.0, 3)) & pose_grid


def test_only_idx_frame_content_matches_index(video):
    frames = list(iter_frames(video, 0.0, None, only_idx={10, 40, 41}))
    assert [f.idx for f in frames] == [10, 40, 41]
    for f in frames:
        assert abs(float(f.img.mean()) - 2 * f.idx) < 6                           # MJPG теряет пару единиц яркости, но кадр узнаётся
