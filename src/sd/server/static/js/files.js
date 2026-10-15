// Скачивание данных, которые уже есть в браузере (CSV, YAML, JSON): без обращения к серверу за файлом.
import { h } from "./dom.js";

export function saveText(name, text, mime = "text/plain;charset=utf-8") {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = h("a", { href: url, download: name, style: { display: "none" } });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/** Ячейка CSV: кавычки удваиваются, значения с запятой, переводом строки или кавычкой заключаются в кавычки; ведущие = + - @ экранируются (защита от формул Excel). */
export function csvCell(v) {
  let t = v === null || v === undefined ? "" : String(v);
  if (/^[=+\-@]/.test(t) && !/^-?\d+([.,]\d+)?$/.test(t)) t = `'${t}`;
  return /[",\n;]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
}

export const toCsv = (rows) => `﻿${rows.map((r) => r.map(csvCell).join(",")).join("\r\n")}\r\n`;
