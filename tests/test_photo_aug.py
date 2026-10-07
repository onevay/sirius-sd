import numpy as np

from sd import photo_aug as AUG


def _img(h=120, w=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_zoom_out_keeps_size_and_shrinks_the_subject():
    img = np.zeros((100, 100, 3), np.uint8)
    img[40:60, 40:60] = 255                                           # «объект» 20×20 в центре
    out = AUG.zoom_out(img, np.random.default_rng(0), 0.5, 0.5)
    assert out.shape == img.shape
    bright = (out.max(axis=2) > 200)
    ys, xs = np.nonzero(bright)
    assert 8 <= ys.max() - ys.min() + 1 <= 12 and 8 <= xs.max() - xs.min() + 1 <= 12   # объект стал ≈ 10×10


def test_degrade_changes_pixels_but_not_shape_and_is_reproducible():
    img = _img()
    a = AUG.degrade(img, np.random.default_rng(1))
    b = AUG.degrade(img, np.random.default_rng(1))
    assert a.shape == img.shape and a.dtype == np.uint8 and np.array_equal(a, b) and not np.array_equal(a, img)


def test_cctv_like_pipeline_and_keys():
    out = AUG.cctv_like(_img(), np.random.default_rng(3))
    assert out.shape == (120, 100, 3)
    assert AUG.aug_key("data/x.jpg", 2) == "data/x.jpg|aug2"
