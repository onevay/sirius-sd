"""X-CLIP: кадры «трубки» обязаны доходить до модели (в transformers 4.57 `XCLIPProcessor(videos=...)` молча теряет видео, модель получала pixel_values=None).

Тест грузит локальные веса (≈ 8 с); без них — пропускается.
"""
import numpy as np
import pytest

from sd.config import load_config


@pytest.fixture(scope="module")
def scorer():
    from sd.tube import XCLIPScorer

    try:
        return XCLIPScorer(load_config())
    except Exception as e:   # весов нет локально / transformers недоступен
        pytest.skip(f"X-CLIP недоступен: {type(e).__name__}: {e}")


def test_xclip_scores_depend_on_video_frames(scorer):
    rng = np.random.default_rng(0)
    noisy = [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
    black = [np.zeros((224, 224, 3), np.uint8) for _ in range(16)]
    a, b = scorer.score(noisy), scorer.score(black)
    assert len(a) == len(scorer.prompts) == 7
    assert abs(sum(a.values()) - 1.0) < 1e-4 and abs(sum(b.values()) - 1.0) < 1e-4   # softmax по промптам
    assert max(abs(a[k] - b[k]) for k in a) > 1e-3        # разные кадры -> разные вероятности: видео действительно используется
    assert len(scorer.timing) == 2
