"""VLM-верификатор серой зоны (руководство §5.6, шаг 8, §7). Только локальные модели.

Вход — 6–8 кадров кропа человека вокруг кандидата (цикла/события) + строгий промпт с JSON-ответом (`configs/prompts/vlm_prompt.txt`).
Выход — действие (smoking/vaping/drinking/phone/eating/touching_face/other/unclear), видны ли предмет и дым. VLM не определяет время
события — только подтверждает или отклоняет кандидата; ответ «unclear» решение не меняет.
Бэкенды:
  ollama       — локальный Ollama (`ollama serve` на 127.0.0.1): ответ одним токеном (yes/no или буква), числовая оценка «затяжка» по его логитам;
  transformers — локальный запуск (на этом CPU годятся только крошечные модели: SmolVLM2-256M/500M);
  openai       — любой OpenAI-совместимый сервер (llama.cpp `llama-server`, LM Studio, vLLM).
Температура 0, фиксированное число кадров и размер кропа — требование воспроизводимости (в manifest.json пишутся промпт, веса, квантизация).
"""
from __future__ import annotations

from . import _env  # noqa: F401

import base64
import json
import math
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

from .paths import CONFIGS

ACTIONS = ["smoking", "vaping", "drinking", "phone", "eating", "touching_face", "other", "unclear"]
POSITIVE_ACTIONS = {"smoking", "vaping"}


def load_prompt(n_frames: int, seconds: float, path: Path | None = None) -> str:
    txt = (path or CONFIGS / "prompts" / "vlm_prompt.txt").read_text(encoding="utf-8")
    return txt.replace("{n_frames}", str(n_frames)).replace("{seconds:.1f}", f"{seconds:.1f}")


def parse_answer(text: str) -> dict:
    """Достаёт JSON из ответа модели; при неудаче — эвристика по ключевым словам. Всегда возвращает action/object_visible/smoke_visible."""
    out = dict(action="unclear", object_visible=None, smoke_visible=None, puff=None, parsed=False, raw=text)
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            a = str(d.get("action", "unclear")).lower().strip()
            puff = str(d.get("puff", "")).lower().strip()
            out.update(action=a if a in ACTIONS else "unclear", object_visible=d.get("object_visible"), smoke_visible=d.get("smoke_visible"),
                       puff=puff if puff in ("yes", "no") else None, parsed=True)
            return out
        except json.JSONDecodeError:
            pass
    low = text.lower()
    for a in ACTIONS[:-1]:
        if a.replace("_", " ") in low or a in low:
            out["action"] = a
            break
    return out


def smoke_score(ans: dict) -> float:
    """Число для вектора признаков: вероятность «yes» по логитам (Ollama), иначе 1 — smoking/vaping, 0 — другое действие, NaN — unclear (флаг отдельно)."""
    if ans.get("p_puff") is not None:
        return float(ans["p_puff"])
    if ans["action"] == "unclear":
        return float("nan")
    return 1.0 if ans["action"] in POSITIVE_ACTIONS else 0.0


def _to_jpeg_b64(img_rgb: np.ndarray, size: int) -> str:
    img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


class OpenAICompatVLM:
    """Клиент OpenAI-совместимого сервера (llama.cpp / Ollama / LM Studio). Изображения — data-URL, температура 0."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", model: str = "local", size: int = 336, timeout: float = 600.0):
        self.base_url, self.model, self.size, self.timeout = base_url.rstrip("/"), model, size, timeout

    def verify(self, frames_rgb: list[np.ndarray], seconds: float) -> dict:
        import httpx

        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _to_jpeg_b64(f, self.size)}} for f in frames_rgb]
        content.append({"type": "text", "text": load_prompt(len(frames_rgb), seconds)})
        t0 = time.perf_counter()
        r = httpx.post(f"{self.base_url}/chat/completions", timeout=self.timeout,
                       json=dict(model=self.model, temperature=0, max_tokens=80, messages=[{"role": "user", "content": content}]))
        r.raise_for_status()
        ans = parse_answer(r.json()["choices"][0]["message"]["content"])
        ans["ms"] = (time.perf_counter() - t0) * 1000
        return ans


OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {"puff": {"type": "string", "enum": ["yes", "no"]}, "action": {"type": "string", "enum": ACTIONS}, "object_visible": {"type": "boolean"}},
    "required": ["puff", "action", "object_visible"],
}
# вариант ответа «буквой» -> действие (порядок как в configs/prompts/vlm_letter.txt)
LETTER_ACTIONS = {"A": "smoking", "B": "vaping", "C": "drinking", "D": "phone", "E": "eating", "F": "touching_face", "G": "other"}
MODE_PROMPTS = {"yesno": "vlm_yesno.txt", "letter": "vlm_letter.txt", "json": "vlm_prompt.txt"}


def first_token_scores(top_logprobs: list[dict] | None, mode: str) -> tuple[float | None, dict[str, float]]:
    """Вероятности по логитам ПЕРВОГО сгенерированного токена (он и есть ответ: yes/no или буква). Возвращает (p_puff, распределение).

    yesno : p_puff = P(yes) / (P(yes) + P(no)); распределение {"yes": p, "no": p};
    letter: p_puff = (P(A) + P(B)) / сумма по буквам A..G (курение + вейп); распределение по буквам.
    Токены сравниваются без регистра, пробелов и точки: «Yes», « yes», «A.» считаются.
    """
    if not top_logprobs:
        return None, {}
    mass: dict[str, float] = {}
    for a in top_logprobs:
        t = str(a.get("token", "")).strip().rstrip(".:").strip().lower()
        if mode == "yesno":
            key = "yes" if t in ("yes", "y") else "no" if t in ("no", "n") else None
        else:
            key = t.upper() if t.upper() in LETTER_ACTIONS else None
        if key:
            mass[key] = mass.get(key, 0.0) + math.exp(a["logprob"])
    total = sum(mass.values())
    if total <= 0:
        return None, {}
    dist = {k: v / total for k, v in mass.items()}
    p = dist.get("yes", 0.0) if mode == "yesno" else dist.get("A", 0.0) + dist.get("B", 0.0)
    return p, dist


class OllamaVLM:
    """Локальный Ollama (`ollama serve`, 127.0.0.1:11434): нативный /api/chat с картинками, thinking выключен, температура 0.

    Режимы: `yesno` и `letter` — модель отвечает ОДНИМ токеном, оценка берётся из его логитов (непрерывная, пригодна для AUC; Ollama отдаёт логиты
    только без `format`); `json` — ответ по JSON-схеме (действие, предмет) без вероятностей, для диагностики. При точно таком же повторном запросе
    сервер берёт всё из кэша (prefill 0.3 с вместо ≈ 10), но при другом тексте вопроса кэш не помогает: каждый режим на окне стоит полный prefill."""

    def __init__(self, model: str = "qwen3.5:2b-q4_K_M", base_url: str = os.environ.get("SD_OLLAMA_URL", "http://127.0.0.1:11434"), size: int = 224, mode: str = "yesno", num_ctx: int = 4096,
                 keep_alive: str = "30m", timeout: float = 900.0, prompt_file: Path | None = None, top_logprobs: int = 10, num_thread: int | None = None):
        assert mode in MODE_PROMPTS, f"mode ∈ {list(MODE_PROMPTS)}"
        self.model, self.base_url, self.size, self.mode, self.num_ctx = model, base_url.rstrip("/"), size, mode, num_ctx
        self.keep_alive, self.timeout, self.top_logprobs, self.num_thread = keep_alive, timeout, top_logprobs, num_thread
        self.prompt_file = prompt_file or CONFIGS / "prompts" / MODE_PROMPTS[mode]

    def verify(self, frames_rgb: list[np.ndarray], seconds: float) -> dict:
        import httpx

        images = [_to_jpeg_b64(f, self.size) for f in frames_rgb]
        opts = dict(temperature=0, seed=0, num_ctx=self.num_ctx, num_predict=48 if self.mode == "json" else 1)
        if self.num_thread:
            opts["num_thread"] = self.num_thread
        # ВАЖНО: кадры и вопрос — в ОДНОМ сообщении. Если разнести их по двум, Ollama берёт картинки только из последнего сообщения (проверено: модель
        # получала 73 токена вместо ≈ 400 и отвечала одинаково на разные окна). Поэтому и кэш префикса между разными вопросами не работает.
        messages = [dict(role="user", content=load_prompt(len(images), seconds, self.prompt_file), images=images)]
        payload = dict(model=self.model, stream=False, think=False, keep_alive=self.keep_alive, options=opts, messages=messages)
        if self.mode == "json":
            payload["format"] = OLLAMA_SCHEMA
        else:
            payload.update(logprobs=True, top_logprobs=self.top_logprobs)
        t0 = time.perf_counter()
        r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama ответил {e.response.status_code}: {e.response.text[:300]} — если это нехватка памяти (Vulkan OutOfDeviceMemory), "
                               f"перезапустите сервер без --igpu (sd.cmd ollama down; sd.cmd ollama up) или возьмите модель с квантизацией q4_K_M") from e
        j = r.json()
        wall = (time.perf_counter() - t0) * 1000
        text = j.get("message", {}).get("content", "")
        if self.mode == "json":
            ans, p, dist = parse_answer(text), None, {}
        else:
            lp = j.get("logprobs") or []
            p, dist = first_token_scores(lp[0].get("top_logprobs") if lp else None, self.mode)
            if self.mode == "yesno":
                action = "unclear" if p is None else "smoking" if p >= 0.5 else "other"
            else:
                action = LETTER_ACTIONS.get(max(dist, key=dist.get), "unclear") if dist else "unclear"
            ans = dict(action=action, object_visible=None, smoke_visible=None, puff=None if p is None else ("yes" if p >= 0.5 else "no"), parsed=p is not None, raw=text)
        ns = 1e6   # длительности Ollama — в наносекундах
        ans.update(ms=wall, p_puff=p, dist=dist, prompt_tokens=j.get("prompt_eval_count"), cached_tokens=j.get("prompt_eval_cached_count"), new_tokens=j.get("eval_count"),
                   prefill_ms=j.get("prompt_eval_duration", 0) / ns, decode_ms=j.get("eval_duration", 0) / ns, load_ms=j.get("load_duration", 0) / ns)
        return ans


class TransformersVLM:
    """Локальный запуск через transformers (CPU, fp32). Для этого ноутбука годятся только крошечные модели (SmolVLM2-256M/500M)."""

    def __init__(self, repo: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", size: int = 336, max_new_tokens: int = 60):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.size, self.max_new_tokens = torch, size, max_new_tokens
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(repo)
        try:  # без нарезки изображения на тайлы — иначе токенов в разы больше
            self.processor.image_processor.do_image_splitting = False
            self.processor.image_processor.size = {"longest_edge": size}
        except Exception:
            pass
        self.model = AutoModelForImageTextToText.from_pretrained(repo, torch_dtype=torch.float32).eval()
        self.load_s = time.perf_counter() - t0

    def verify(self, frames_rgb: list[np.ndarray], seconds: float) -> dict:
        from PIL import Image

        imgs = [Image.fromarray(cv2.resize(f, (self.size, self.size), interpolation=cv2.INTER_AREA)) for f in frames_rgb]
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": load_prompt(len(imgs), seconds)}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.processor(text=text, images=imgs, return_tensors="pt")
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False)
        dt = (time.perf_counter() - t0) * 1000
        gen = self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        ans = parse_answer(gen)
        ans.update(ms=dt, prompt_tokens=int(inp["input_ids"].shape[1]), new_tokens=int(out.shape[1] - inp["input_ids"].shape[1]))
        return ans
