"""VLM: клиент Ollama (запрос, режимы yes/no и «буква», оценка по логитам первого токена) и сводка качества на размеченных циклах."""
import math

import numpy as np
import pandas as pd
import pytest

from sd import vlm
from sd import vlm_eval as VE


def _top(*pairs):
    return [{"token": t, "logprob": math.log(p)} for t, p in pairs]


def test_yesno_score_normalises_yes_against_no_and_ignores_case_and_spaces():
    p, dist = vlm.first_token_scores(_top(("Yes", 0.45), (" yes", 0.15), ("no", 0.2), ("maybe", 0.2)), "yesno")
    assert abs(p - 0.75) < 1e-9 and abs(dist["yes"] - 0.75) < 1e-9 and abs(dist["no"] - 0.25) < 1e-9   # 0.6 / (0.6 + 0.2): посторонние токены не учитываются


def test_letter_score_is_smoking_plus_vaping_share():
    p, dist = vlm.first_token_scores(_top(("A", 0.3), ("B.", 0.1), ("C", 0.4), (" F", 0.1), ("The", 0.1)), "letter")
    assert abs(p - 0.4 / 0.9) < 1e-9 and set(dist) == {"A", "B", "C", "F"}


def test_scores_none_without_logprobs_or_matching_tokens():
    assert vlm.first_token_scores(None, "yesno") == (None, {}) and vlm.first_token_scores([], "letter") == (None, {})
    assert vlm.first_token_scores(_top(("hello", 0.9)), "yesno") == (None, {})


def test_parse_answer_reads_puff_field():
    a = vlm.parse_answer('{"puff": "yes", "action": "smoking", "object_visible": true}')
    assert a["puff"] == "yes" and a["action"] == "smoking" and a["parsed"]
    assert vlm.parse_answer('{"puff": "maybe", "action": "other"}')["puff"] is None


def test_smoke_score_prefers_logit_probability():
    assert vlm.smoke_score(dict(action="drinking", p_puff=0.83)) == 0.83
    assert vlm.smoke_score(dict(action="smoking")) == 1.0 and vlm.smoke_score(dict(action="drinking")) == 0.0
    assert math.isnan(vlm.smoke_score(dict(action="unclear")))


class _Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def _fake_post(monkeypatch, body):
    captured = {}

    def post(url, json=None, timeout=None):
        captured.update(url=url, json=json, timeout=timeout)
        return _Resp(body)

    import httpx

    monkeypatch.setattr(httpx, "post", post)
    return captured


FRAMES = [np.zeros((224, 224, 3), np.uint8) for _ in range(6)]
TIMING = {"prompt_eval_count": 540, "prompt_eval_cached_count": 0, "eval_count": 1, "prompt_eval_duration": 12_000_000_000, "eval_duration": 150_000_000, "load_duration": 3_000_000}


def test_ollama_yesno_request_and_response(monkeypatch):
    body = {"message": {"content": "no"}, "logprobs": [{"token": "no", "logprob": math.log(0.7), "top_logprobs": _top(("no", 0.7), ("yes", 0.1), ("No", 0.1))}], **TIMING}
    cap = _fake_post(monkeypatch, body)
    ans = vlm.OllamaVLM("qwen3.5:2b").verify(FRAMES, 2.5)
    j = cap["json"]
    assert cap["url"] == "http://127.0.0.1:11434/api/chat"                      # только локальный сервер
    assert j["model"] == "qwen3.5:2b" and j["think"] is False and j["stream"] is False
    msgs = j["messages"]
    assert len(msgs) == 1 and len(msgs[0]["images"]) == 6        # кадры и вопрос в ОДНОМ сообщении: Ollama берёт картинки только из последнего (проверено на сервере)
    assert "format" not in j                                                      # с `format` Ollama не отдаёт логиты по всем токенам
    assert j["logprobs"] is True and j["options"]["num_predict"] == 1 and j["options"]["temperature"] == 0 and j["options"]["seed"] == 0
    assert "6 frames" in msgs[0]["content"] and "2.5 seconds" in msgs[0]["content"] and "yes or no" in msgs[0]["content"]
    assert ans["action"] == "other" and abs(ans["p_puff"] - 0.1 / 0.9) < 1e-9 and ans["puff"] == "no"
    assert ans["prompt_tokens"] == 540 and ans["cached_tokens"] == 0 and abs(ans["prefill_ms"] - 12000) < 1e-6 and abs(ans["decode_ms"] - 150) < 1e-6


def test_ollama_letter_mode_returns_action_and_smoking_probability(monkeypatch):
    body = {"message": {"content": "C"}, "logprobs": [{"token": "C", "logprob": math.log(0.6), "top_logprobs": _top(("C", 0.6), ("A", 0.3), ("B", 0.1))}], **TIMING}
    cap = _fake_post(monkeypatch, body)
    ans = vlm.OllamaVLM("m", mode="letter").verify(FRAMES, 2.0)
    assert "A. smoking a cigarette" in cap["json"]["messages"][0]["content"]
    assert ans["action"] == "drinking" and abs(ans["p_puff"] - 0.4) < 1e-9


def test_ollama_json_mode_uses_schema_and_has_no_probability(monkeypatch):
    body = {"message": {"content": '{"puff": "yes", "action": "smoking", "object_visible": true}'}, **TIMING}
    cap = _fake_post(monkeypatch, body)
    ans = vlm.OllamaVLM("m", mode="json").verify(FRAMES, 2.0)
    assert cap["json"]["format"] == vlm.OLLAMA_SCHEMA and "logprobs" not in cap["json"] and cap["json"]["options"]["num_predict"] == 48
    assert len(cap["json"]["messages"]) == 1                                      # у json-промпта нет разделителя: одно сообщение
    assert ans["action"] == "smoking" and ans["p_puff"] is None and ans["object_visible"] is True


def test_ollama_unknown_mode_is_rejected():
    with pytest.raises(AssertionError):
        vlm.OllamaVLM("m", mode="chatty")


def test_pick_subset_is_balanced_and_spreads_over_videos():
    rows = [dict(video=f"v{i % 4}", y=int(i % 2 == 0), label="smoke" if i % 2 == 0 else "drink", start=float(i)) for i in range(40)]
    tab = pd.DataFrame(rows)
    sub = VE.pick_subset(tab, 6, seed=1)
    assert sub.y.sum() == 6 and (1 - sub.y).sum() == 6
    assert sub[sub.y == 1].video.nunique() >= 2                                # по кругу по видео, не всё из одного клипа
    assert VE.pick_subset(tab, 6, seed=1).index.tolist() == sub.index.tolist()  # детерминированно
    assert len(VE.pick_subset(tab, None)) == 40


def _tab_and_results(n=24, separable=True, modes=("yesno", "letter")):
    rng = np.random.default_rng(0)
    rows, res = [], []
    for i in range(n):
        y = int(i % 2 == 0)
        rows.append(dict(video=f"v{i % 6}", tid=1, start=float(i), y=y, label="smoke" if y else "drink"))
        r = dict(video=f"v{i % 6}", tid=1, start=float(i), ok=True)
        for k, name in enumerate(modes):
            r[f"score_{name}"] = (0.8 if y else 0.2) + rng.normal(0, .05) if (separable and k == 0) else 1.0
            r[f"action_{name}"] = ("smoking" if y else "drinking") if separable else "smoking"
            r[f"ms_{name}"] = 10_000.0 + i if k == 0 else 300.0
            r[f"prefill_ms_{name}"] = 8000.0 if k == 0 else 200.0
            r[f"parsed_{name}"] = True
        r.update(score=r[f"score_{modes[0]}"], action=r[f"action_{modes[0]}"], ms=r[f"ms_{modes[0]}"], parsed=True)
        res.append(r)
    res.append(dict(video="v0", tid=9, start=99.0, ok=False))                  # окно без трека: считается как сбой, а не как ответ
    return pd.DataFrame(rows), pd.DataFrame(res)


def test_evaluate_reports_auc_decision_and_cost_per_mode():
    tab, res = _tab_and_results()
    assert VE.modes_in(res) == ["yesno", "letter"]
    ev = VE.evaluate(tab, res, n_boot=100, mode="yesno")
    assert ev["n"] == 24 and ev["n_failed"] == 1 and ev["auc"] > 0.99
    assert ev["recall"] == 1.0 and ev["fp_rate"] == 0.0 and ev["accuracy"] == 1.0 and abs(ev["answer_yes_share"] - 0.5) < 1e-9
    assert ev["youden_j"] == 1.0 and 0.2 < ev["youden_threshold"] < 0.9
    assert ev["ms_median"] > 10_000 and ev["prefill_ms_median"] == 8000.0 and ev["actions"] == {"smoking": 12, "drinking": 12}
    cheap = VE.evaluate(tab, res, n_boot=50, mode="letter")                     # второй режим на том же окне — из кэша кадров, стоит копейки
    assert cheap["ms_median"] == 300.0


def test_evaluate_constant_yes_has_no_separation():
    tab, res = _tab_and_results(separable=False)
    ev = VE.evaluate(tab, res, n_boot=50, mode="yesno")
    assert ev["answer_yes_share"] == 1.0 and ev["fp_rate"] == 1.0 and ev["recall"] == 1.0
    assert np.isnan(ev["auc"]) or ev["auc"] == 0.5                              # постоянный ответ различающей способности не даёт


def test_ollama_http_error_message_explains_memory_problem(monkeypatch):
    import httpx

    def post(url, json=None, timeout=None):
        return httpx.Response(500, text='{"error":"vk::Device::allocateMemory: ErrorOutOfDeviceMemory"}', request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(RuntimeError, match="OutOfDeviceMemory") as e:
        vlm.OllamaVLM("m").verify(FRAMES, 2.5)
    assert "без --igpu" in str(e.value) and "q4_K_M" in str(e.value)
