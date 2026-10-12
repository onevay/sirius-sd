"""PDF-отчёт прогона оценки (`sd report <прогон>`, кнопка на вкладке «Валидация»).

Собирается только из сохранённого прогона (outputs/experiments/<id>) и кэша — модели не запускаются, отчёт воспроизводим. Вёрстка — matplotlib (PdfPages, A4,
шрифт DejaVu Sans из поставки matplotlib — кириллица без системных зависимостей и без новых пакетов).

Разделы: 1 конфигурация и отпечатки · 2 данные · 3 качество (F1, P, R с интервалом, FP/ч, задержка, бюджет ошибок, замечания) · 4 порог (кривая и плато) ·
5 ошибки по причинам · 6 клипы · 7 производительность (× реального времени по клипам, время этапов, поза кадров/с) · 8 сравнение с базовым прогоном (парный
бутстрэп) · 9 слабые места (правила `cache_analysis.RULES`) · приложение: профиль целиком, git, команда воспроизведения.
"""
from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache_analysis as CA
from . import experiments as XP

A4 = (8.27, 11.69)
INK, MUTED, ACC, RED, ORANGE, GREEN, BLUE = "#1f2937", "#6b7280", "#0f766e", "#b91c1c", "#d97706", "#15803d", "#2563eb"
LEFT, RIGHT, TOP, BOTTOM = 0.07, 0.93, 0.95, 0.06


class _Doc:
    """Простая вёрстка текста и таблиц по страницам A4 (координаты фигуры 0..1)."""

    def __init__(self, pdf, title: str):
        self.pdf, self.title, self.page_no, self.fig, self.y = pdf, title, 0, None, TOP

    def new_page(self) -> None:
        import matplotlib.pyplot as plt

        self.finish()
        self.fig = plt.figure(figsize=A4)
        self.page_no += 1
        self.fig.text(LEFT, 0.975, self.title, fontsize=7.5, color=MUTED)
        self.fig.text(RIGHT, 0.975, f"стр. {self.page_no}", fontsize=7.5, color=MUTED, ha="right")
        self.y = TOP

    def finish(self) -> None:
        import matplotlib.pyplot as plt

        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None

    def need(self, h: float) -> None:
        if self.fig is None or self.y - h < BOTTOM:
            self.new_page()

    def h1(self, text: str) -> None:
        self.need(0.16)                       # заголовок не остаётся один внизу страницы
        self.fig.text(LEFT, self.y - 0.02, text, fontsize=15, color=ACC, weight="bold")
        self.fig.add_artist(__import__("matplotlib").lines.Line2D([LEFT, RIGHT], [self.y - 0.03, self.y - 0.03], color="#99d5cc", lw=1))
        self.y -= 0.05

    def h2(self, text: str) -> None:
        self.need(0.09)
        self.fig.text(LEFT, self.y - 0.015, text, fontsize=11, color=INK, weight="bold")
        self.y -= 0.03

    def para(self, text: str, size: float = 8.8, color: str = INK, width: int | None = None) -> None:
        width = width or int(100 * 8.8 / size)
        for line in textwrap.wrap(text, width=width) or [""]:
            self.need(0.016)
            self.fig.text(LEFT, self.y - 0.012, line, fontsize=size, color=color)
            self.y -= 0.0155
        self.y -= 0.004

    def bullets(self, items: list[str], color: str = INK) -> None:
        for it in items:
            lines = textwrap.wrap(it, width=98) or [""]
            for k, line in enumerate(lines):
                self.need(0.016)
                self.fig.text(LEFT + (0.0 if k == 0 else 0.018), self.y - 0.012, ("• " if k == 0 else "") + line, fontsize=8.6, color=color)
                self.y -= 0.0152
        self.y -= 0.004

    def kv(self, rows: list[tuple[str, str]], key_w: float = 0.27) -> None:
        for k, v in rows:
            lines = textwrap.wrap(str(v), width=68) or ["—"]
            self.need(0.016 * len(lines))
            self.fig.text(LEFT, self.y - 0.012, k, fontsize=8.6, color=MUTED)
            for i, line in enumerate(lines):
                self.fig.text(LEFT + key_w, self.y - 0.012 - 0.0152 * i, line, fontsize=8.6, color=INK)
            self.y -= 0.0152 * len(lines) + 0.002

    def table(self, df: pd.DataFrame, widths: list[float] | None = None, size: float = 7.6, max_chars: int | None = None, colors: dict | None = None) -> None:
        """Таблица с переносом по страницам (шапка повторяется). `colors` — {(строка, колонка): цвет}."""
        if df is None or df.empty:
            self.para("— нет данных —", color=MUTED)
            return
        cols = list(df.columns)
        widths = widths or [1.0 / len(cols)] * len(cols)
        tot = sum(widths)
        widths = [w / tot * (RIGHT - LEFT) for w in widths]
        rh = 0.0165

        def header() -> None:
            self.need(rh * 2)
            self.fig.add_artist(__import__("matplotlib").patches.Rectangle((LEFT, self.y - rh), RIGHT - LEFT, rh, color="#eef6f4", transform=self.fig.transFigure, zorder=0))
            x = LEFT
            for c, w in zip(cols, widths):
                self.fig.text(x + 0.004, self.y - rh * 0.72, str(c), fontsize=size, color=INK, weight="bold")
                x += w
            self.y -= rh

        header()
        for i, row in enumerate(df.itertuples(index=False)):
            if self.y - rh < BOTTOM:
                self.new_page()
                header()
            x = LEFT
            for j, (v, w) in enumerate(zip(row, widths)):
                s = _fmt(v)
                lim = max_chars or max(4, int(w / 0.0068 * 7.6 / size))
                if len(s) > lim:
                    s = s[: lim - 1] + "…"
                self.fig.text(x + 0.004, self.y - rh * 0.72, s, fontsize=size, color=(colors or {}).get((i, j), INK))
                x += w
            self.fig.add_artist(__import__("matplotlib").lines.Line2D([LEFT, RIGHT], [self.y - rh, self.y - rh], color="#e5e7eb", lw=0.5))
            self.y -= rh
        self.y -= 0.008

    def pre(self, lines: list[str], size: float = 6.8) -> None:
        """Моноширинный блок (JSON, команды) с плотным интервалом."""
        for line in lines:
            for part in textwrap.wrap(line, width=int(116 * 6.8 / size), subsequent_indent="    ") or [""]:
                self.need(0.011)
                self.fig.text(LEFT, self.y - 0.009, part, fontsize=size, color=MUTED, family="monospace")
                self.y -= 0.0108
        self.y -= 0.006

    def figure(self, draw, h: float = 0.28, left: float = 0.05) -> None:
        """Вставить график высотой h (доля страницы): draw(ax); `left` — отступ слева под подписи оси (для горизонтальных столбцов — больше)."""
        self.need(h + 0.01)
        ax = self.fig.add_axes([LEFT + left, self.y - h + 0.03, RIGHT - LEFT - left - 0.02, h - 0.045])
        draw(ax)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=MUTED)
        self.y -= h + 0.005


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "да" if v else "нет"
    if isinstance(v, (float, np.floating)):
        return f"{v:.3f}" if abs(v) < 1 else (f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}")
    return str(v)


def _ci(rep) -> str:
    ci = rep.ci.get("f1") or (None, None)
    return f"{ci[0]:.2f}–{ci[1]:.2f}" if ci and ci[0] is not None else "мало клипов"


def reproduce_command(meta: dict) -> str:
    """Команда `sd eval`, повторяющая прогон: профиль по имени + все его параметры и опции явными правками (JSON — подмножество YAML, `-s` их понимает)."""
    prof = meta.get("profile") or {}
    sets = [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (prof.get("config") or {}).items()]
    sets += [f"options.{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (prof.get("options") or {}).items()]
    parts = ["python -m sd eval"] + [f'-d "{x}"' for x in meta.get("dirs", [])] + [f"-p {prof.get('name') or meta.get('name')}"] + [f"-s '{x}'" for x in sets]
    parts += [f"--mode {meta.get('mode', 'events')}", f"--role {meta.get('role', 'validation')}", f"--policy {meta.get('policy', 'fixed')}"]
    if meta.get("max_sec"):
        parts.append(f"--max-sec {meta['max_sec']}")
    return " ".join(parts)


def build(rid: str, out: str | Path | None = None, baseline: str | None = None, root: Path | None = None, author: str = "") -> Path:
    """Собрать PDF прогона `rid` (и сравнение с `baseline`, если задан). Возвращает путь к файлу (по умолчанию — <прогон>/report_<прогон>.pdf)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    from . import gt as GT
    from . import start_eval as SE

    base = (root or XP.EXP_DIR) / rid
    rep, meta, ev, gt = XP.load_run(rid, root)
    m, b, st = rep.metrics, rep.budget, rep.settings
    tab = CA.run_table(rid, root)
    try:
        gest = pd.read_csv(SE.GT_CSV, encoding="utf-8-sig")
    except Exception:       # noqa: BLE001
        gest = None
    spots = CA.weak_spots(tab, rep.errors, gest)
    out = Path(out) if out else base / f"report_{rid}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    d = meta.get("describe", {})
    with PdfPages(out) as pdf:
        doc = _Doc(pdf, f"Отчёт о проверке конфигурации · {meta.get('name')} · {rid}")
        info = pdf.infodict()
        info["Title"], info["Author"], info["Subject"] = f"Отчёт прогона {rid}", author or "Dev Studio", "Event F1, ошибки, производительность"
        # --- титул
        doc.new_page()
        doc.fig.text(LEFT, 0.80, "Отчёт о проверке конфигурации", fontsize=24, color=ACC, weight="bold")
        doc.fig.text(LEFT, 0.765, f"профиль «{meta.get('name')}» · прогон {rid}", fontsize=12, color=INK)
        role = {"validation": "валидация (порог можно было подбирать)", "hidden": "скрытый набор (порог заморожен)"}.get(meta.get("role"), meta.get("role"))
        doc.fig.text(LEFT, 0.74, f"{role} · эталон: {'ручная разметка событий' if rep.mode == 'events' else 'по именам папок (грубо)'}", fontsize=10, color=MUTED)
        big = [("Event F1", f"{m['f1']:.3f}", f"95 %: {_ci(rep)}"), ("Precision", f"{m['precision']:.2f}", f"FP {m['fp']}"), ("Recall", f"{m['recall']:.2f}", f"FN {m['fn']}"),
               ("Порог", f"{m['threshold']:.2f}", meta.get("policy", ""))]
        for i, (k, v, sub) in enumerate(big):
            x = LEFT + i * 0.215
            doc.fig.text(x, 0.66, k, fontsize=9, color=MUTED)
            doc.fig.text(x, 0.625, v, fontsize=22, color=ACC if i == 0 else INK, weight="bold")
            doc.fig.text(x, 0.605, sub, fontsize=8, color=MUTED)
        doc.y = 0.56
        warn = []
        leak = tab.leak.sum() if "leak" in tab else 0
        if leak:
            warn.append(f"Классификатор цикла обучен на {int(leak)} из {len(tab)} клипов этого прогона — цифра оптимистична; честную даёт вложенная oof-оценка.")
        failed = [c for c in meta.get("clips", []) if c.get("error")]
        if failed:
            warn.append(f"Клипов с ошибкой обработки: {len(failed)} — посчитаны как «событий нет».")
        if meta.get("max_sec"):
            warn.append(f"Оценивались только первые {meta['max_sec']} с каждого клипа.")
        warn += list(rep.notes)
        if warn:
            doc.h2("Важно знать до чтения цифр")
            doc.bullets(warn, color=RED if leak or failed else INK)
        doc.h2("Кратко")
        doc.kv([("клипов / часов видео", f"{m.get('clips')} / {m.get('hours', 0):.2f} ч"), ("эпизодов в эталоне", f"{m.get('n_gt')} (игнорировано предсказаний: {m.get('ignored', 0)})"),
                ("ложных в час", f"{_fmt(m.get('fp_per_hour'))} на клипах без курения; {_fmt(m.get('fp_per_hour_all'))} по всем"),
                ("медианная задержка", f"{_fmt(m.get('median_latency'))} с"), ("цель F1", f"{b.get('target')} — {'достигнута' if b.get('reached') else 'не достигнута'}; ошибок {b.get('errors')} из {b.get('allowed'):.0f} допустимых"),
                ("собран", time.strftime("%Y-%m-%d %H:%M") + (f" · {author}" if author else ""))])
        # --- 1. конфигурация
        doc.h1("1. Конфигурация и отпечатки")
        doc.kv([(k, str(v)) for k, v in d.items()] + [("отпечаток профиля", meta.get("fingerprint")), ("отпечаток эталона", meta.get("gt_fingerprint") or "—"),
                                                      ("git", f"{(meta.get('git') or {}).get('commit')}{' (есть незакоммиченные правки)' if (meta.get('git') or {}).get('dirty') else ''}"),
                                                      ("создан", meta.get("created"))])
        host = meta.get("host")
        doc.h2("Машина прогона")
        doc.kv([(k, str(v)) for k, v in host.items()] if host else [("—", "в прогоне не записано (прогон старше Dev Studio); скорость ниже — по кэшу этой машины")])
        # --- 2. данные
        doc.h1("2. Данные")
        doc.kv([("папки", ", ".join(Path(x).name for x in meta.get("dirs", []))), ("клипов в оценке", str(len(meta.get("clips", [])))),
                ("пропущено неразмеченных", str(len(meta.get("skipped_unlabeled", [])))), ("строк эталона", str(meta.get("gt_rows")))])
        if len(gt) and "label" in gt:
            doc.table(gt.label.value_counts().rename_axis("метка").reset_index(name="строк"), [2, 1])
        if "h_med" in tab and tab.h_med.notna().any():
            h = tab.h_med.dropna()
            doc.para(f"Рост людей в кадре (медиана по трекам, по клипам): мин {h.min():.0f} px, медиана {h.median():.0f} px, макс {h.max():.0f} px. "
                     f"Клипов, где люди мельче 150 px: {(h < 150).sum()}. Ниже ≈ 100 px поза ненадёжна.", color=MUTED)
        # --- 3. порог
        doc.h1("3. Порог: кривая F1, precision, recall")
        cur = rep.curve
        if len(cur):
            def draw_curve(ax):
                pl = rep.plateau
                if pl.get("lo") is not None:
                    ax.axvspan(pl["lo"], pl["hi"], color=BLUE, alpha=0.08)
                ax.plot(cur.threshold, cur.f1, color=ACC, lw=2.2, label="F1")
                ax.plot(cur.threshold, cur.precision, color=BLUE, lw=1.2, label="precision")
                ax.plot(cur.threshold, cur.recall, color=ORANGE, lw=1.2, label="recall")
                ax.axvline(st["threshold"], color=RED, lw=1.2)
                ax.axhline(st.get("target_f1", 0.8), color=MUTED, lw=0.8, ls="--")
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("порог уверенности события", fontsize=8)
                ax.legend(fontsize=7.5, frameon=False, loc="lower left")
            doc.figure(draw_curve, 0.3)
            pl = rep.plateau
            if pl.get("center") is not None:
                doc.para(f"Лучший F1 {pl['best_f1']:.3f} при пороге {pl['best_threshold']:.2f}; плато (в пределах {pl['tol']:.2f} от лучшего) {pl['lo']:.2f}–{pl['hi']:.2f}. "
                         f"Порог прогона {st['threshold']:.2f} (политика «{meta.get('policy')}»). Узкое плато — порог хрупкий.", color=MUTED)
        # --- 4. ошибки
        doc.h1("4. Ошибки по причинам")
        er = rep.errors
        if len(er):
            cnt = er.groupby(["kind", "cause_ru"]).size().reset_index(name="n").sort_values("n")

            def draw_err(ax):
                ax.barh(range(len(cnt)), cnt.n, color=[RED if k == "FP" else ORANGE for k in cnt.kind])
                ax.set_yticks(range(len(cnt)))
                ax.set_yticklabels([textwrap.shorten(f"{k}: {c}", 58, placeholder="…") for k, c in zip(cnt.kind, cnt.cause_ru)], fontsize=7)
                ax.set_xlabel("число", fontsize=8)
                ax.xaxis.get_major_locator().set_params(integer=True)
            doc.figure(draw_err, min(0.08 + 0.035 * len(cnt), 0.3), left=0.42)
            show = er[["kind", "clip_id", "start", "end", "confidence", "cause_ru"]].rename(columns={"kind": "тип", "clip_id": "клип", "start": "начало", "end": "конец",
                                                                                                  "confidence": "увер.", "cause_ru": "причина"})
            doc.table(show.round(2), [0.5, 2.4, 0.7, 0.7, 0.6, 3.4], size=7.2)
        else:
            doc.para("Ошибок нет.")
        # --- 5. клипы
        doc.h1("5. По клипам")
        pc = tab[[c for c in ("clip_id", "duration", "gt_pos", "pred", "tp", "fp", "fn", "f1", "status", "leak") if c in tab]].copy()
        pc = pc.sort_values(["f1", "fn"], ascending=[True, False], na_position="last") if "f1" in pc else pc
        colors = {}
        for i, r in enumerate(pc.itertuples(index=False)):
            st_ = str(getattr(r, "status", ""))
            if st_ not in ("ok", "nan", ""):
                colors[(i, list(pc.columns).index("status"))] = RED if "FN" in st_ else ORANGE
        doc.table(pc.rename(columns={"clip_id": "клип", "duration": "длит., с", "gt_pos": "эталон", "pred": "найдено", "status": "статус", "leak": "в обучении"}),
                  [3, 0.8, 0.7, 0.7, 0.5, 0.5, 0.5, 0.6, 1.0, 0.8], colors=colors)
        # --- 6. производительность
        doc.h1("6. Производительность")
        doc.para("× реального времени = длительность клипа / время вычислений по замерам кэша (поза — время её расчёта, остальное — этапы последнего распознавания). "
                 "≥ 1 — конвейер успевает за видео. Время самого прогона оценки не показательно: поза и признаки берутся из кэша.", color=MUTED)
        if "rt_factor" in tab and tab.rt_factor.notna().any():
            rt = tab.dropna(subset=["rt_factor"]).sort_values("rt_factor")

            def draw_rt(ax):
                ax.barh(range(len(rt)), rt.rt_factor, color=[GREEN if v >= 1 else ORANGE for v in rt.rt_factor])
                ax.axvline(1.0, color=RED, lw=1)
                ax.set_yticks(range(len(rt)))
                ax.set_yticklabels([textwrap.shorten(c, 34) for c in rt.clip_id], fontsize=6.2)
                ax.set_xlabel("× реального времени", fontsize=8)
            doc.figure(draw_rt, min(0.06 + 0.0125 * len(rt), 0.55), left=0.25)
            q = tab.rt_factor.dropna()
            doc.para(f"Медиана {q.median():.2f}×, худший {q.min():.2f}× ({tab.loc[tab.rt_factor.idxmin(), 'clip_id']}); быстрее реального времени {int((q >= 1).sum())} из {len(q)} клипов.")
        tcols = [c for c in tab.columns if c.startswith("t_") and c not in ("t_total",)]
        if tcols:
            tt = tab[tcols].sum(numeric_only=True).sort_values(ascending=False)
            tt = tt[tt > 0]
            doc.h2("Время этапов (сумма по клипам, последнее распознавание, с)")
            doc.table(pd.DataFrame(dict(этап=[c[2:] for c in tt.index], секунд=tt.round(1).to_numpy())), [3, 1])
            doc.para("Этапы с суффиксом _cached — чтение готового из кэша; поза здесь — загрузка кэша, её реальное время — в колонке «поза, к/с» ниже.", color=MUTED)
        perf = tab[[c for c in ("clip_id", "width", "height", "proc_fps", "pose_fps", "pose_ms", "rt_factor") if c in tab]].rename(
            columns={"clip_id": "клип", "width": "ширина", "height": "высота", "proc_fps": "к/с нужно", "pose_fps": "поза, к/с", "pose_ms": "мс/кадр", "rt_factor": "× р.в."})
        doc.table(perf, [3, 0.7, 0.7, 0.8, 0.8, 0.8, 0.7])
        # --- 7. сравнение
        if baseline:
            from . import sweep as SW

            doc.h1(f"7. Сравнение с базовым прогоном {baseline}")
            try:
                r = SW.compare_saved(baseline, rid)
                if r.get("diff") is None:
                    doc.para("Мало общих клипов для парного сравнения (нужно ≥ 3).")
                else:
                    verdict = "этот прогон лучше" if r["lo"] > 0 else ("базовый лучше" if r["hi"] < 0 else "различие в пределах шума")
                    doc.kv([("общих клипов", str(r["clips"])), ("F1: этот − базовый", f"{r['diff'] * 100:+.1f} п.п. (95 %: {r['lo'] * 100:+.1f} … {r['hi'] * 100:+.1f})"),
                            ("вероятность «этот лучше»", f"{r['p_better']:.0%}"), ("вывод", verdict), ("эталон одинаковый", "да" if r["same_gt"] else "НЕТ — сравнение некорректно")])
                ra, _, _, _ = XP.load_run(baseline, root)
                a = ra.per_clip[["clip_id", "f1", "status"]].rename(columns={"f1": "F1 базовый", "status": "статус базовый"})
                bb = rep.per_clip[["clip_id", "f1", "status"]].rename(columns={"f1": "F1 этот", "status": "статус этот"})
                j = a.merge(bb, on="clip_id")
                j = j[(j["F1 базовый"].fillna(-1) != j["F1 этот"].fillna(-1)) | (j["статус базовый"] != j["статус этот"])]
                doc.h2("Клипы, где результат изменился")
                doc.table(j.rename(columns={"clip_id": "клип"}), [3, 1, 1.2, 1, 1.2])
            except Exception as e:      # noqa: BLE001
                doc.para(f"сравнение не удалось: {e}", color=RED)
        # --- 8. слабые места
        doc.h1(("8" if baseline else "7") + ". Слабые места (автоматические правила)")
        doc.table(CA.summary(spots), [1.3, 0.8, 0.6, 5])
        major = spots[spots.level != "info"]
        if len(major):
            doc.h2("Ошибки и предупреждения по клипам")
            doc.table(major[["level", "clip_id", "rule", "detail"]].rename(columns={"level": "уровень", "clip_id": "клип", "rule": "правило", "detail": "подробности"}),
                      [0.8, 2.6, 1.3, 3.8], size=7)
        info_n = spots[spots.level == "info"].groupby("rule").clip_id.nunique() if len(spots) else {}
        if len(info_n):
            doc.para("Справочные правила (по клипам не расписаны): " + "; ".join(f"{k} — {v} клипов" for k, v in info_n.items()) + ". Полный список — вкладка «Валидация → Анализ кэша».",
                     color=MUTED)
        # --- приложение
        doc.h1("Приложение: воспроизведение")
        doc.para("Команда воспроизводит конфигурацию прогона точно: параметры профиля передаются правками -s (сам файл профиля мог с тех пор измениться).", color=MUTED)
        doc.pre([reproduce_command(meta)], size=7.2)
        doc.para("Профиль целиком:", size=8, color=MUTED)
        doc.pre(json.dumps(meta.get("profile", {}), ensure_ascii=False, indent=1).splitlines()[:90])
        doc.finish()
    return out
