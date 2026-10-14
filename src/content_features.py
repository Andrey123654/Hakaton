# -*- coding: utf-8 -*-
"""
content_features.py — признаки на основе HTML-контента страниц.

Организаторы выложили расширенный набор (`train_with_content.csv`,
`test_with_content.csv`) с колонкой `content` — сохранённым HTML страницы.
Такие признаки работают значительно сильнее «одного URL»:

* структура страницы: формы, поля пароля, iframe, внешние скрипты, meta refresh,
  JS-редиректы, обфускация (eval, base64, \\x-escape);
* подмена бренда: название бренда в заголовке/тексте, тогда как домен ему
  не принадлежит (классическая мимикрия «копия страницы банка»);
* признаки социальной инженерии: «подтвердите аккаунт», «ваш пароль истекает»;
* признаки парковки домена/заглушки: «domain for sale», «buy this domain»;
* внешние домены в формах и ресурсах (кража учётных данных на чужой хост).

Модуль читает многогигабайтный CSV потоково (chunksize) и распараллеливает
разбор HTML по процессам.

Запуск:
    python src/content_features.py --csv train_with_content.csv \
        --out cache/content_train.pkl --workers 8
    python src/content_features.py --csv test_with_content.csv \
        --out cache/content_test.pkl --workers 8
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import BRANDS, PHISH_WORDS, homoglyph_normalize, normalize_url, split_host  # noqa: E402

# Сколько первых символов HTML анализируем (страницы-клоны обычно короткие,
# а «тяжёлые» легитимные страницы резать без потери сигнала можно).
MAX_HTML = 120_000
MAX_HTML_HARD = 400_000

TAG_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)((?:\s+[^<>]*)?)/?>")
ATTR_RE = re.compile(r"""([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")
TEXT_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.I | re.S)
WS_RE = re.compile(r"\s+")
BASE64_RE = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")
HEXESC_RE = re.compile(r"\\x[0-9a-fA-F]{2}")
EVAL_RE = re.compile(r"\b(eval|unescape|fromCharCode|atob|decodeURIComponent)\s*\(", re.I)
JS_REDIRECT_RE = re.compile(
    r"(window|document|top|self)\.location(\.href|\.replace|\.assign)?\s*=", re.I)
META_REFRESH_RE = re.compile(r"""http-equiv\s*=\s*["']?refresh""", re.I)
META_REFRESH_URL_RE = re.compile(r"""content\s*=\s*["'][^"']*url\s*=\s*([^"';]+)""", re.I)
OBFUSCATED_JS_RE = re.compile(r"charCodeAt|String\.fromCharCode|\\\\u00|%\d\d%", re.I)
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
INPUT_RE = re.compile(r"<input\b[^>]*>", re.I)

# Бренды для сопоставления с текстом страницы (только длинные и однозначные).
CONTENT_BRANDS = [b for b in BRANDS if len(b) >= 5 and "-" not in b]
BRAND_RE = re.compile(r"\b(" + "|".join(sorted(set(CONTENT_BRANDS), key=len, reverse=True)) + r")\b", re.I)

# Фразы социальной инженерии (ищутся в тексте страницы).
PHISH_PHRASES = [
    "verify your account", "verify your identity", "confirm your identity",
    "unusual sign-in", "unusual activity", "suspicious activity",
    "your account has been", "account has been suspended", "account will be closed",
    "update your payment", "update your billing", "payment information",
    "your password", "password will expire", "reset your password",
    "click here to confirm", "sign in to continue", "log in to continue",
    "confirm your account", "reactivate your account", "validate your account",
    "подтвердите", "ваш аккаунт", "ваш пароль", "войдите в",
    "введите данные карты", "номер карты", "cvv", "срок действия карты",
    "войти в личный кабинет", "заблокирован", "восстановление доступа",
]
PARKED_PHRASES = [
    "domain is for sale", "buy this domain", "this domain is parked",
    "domain for sale", "sedoparking", "godaddy", "hugedomains", "afternic",
    "домен продается", "домен выставлен на продажу", "parked free",
]
LOGIN_HINTS = [
    "sign in", "log in", "login", "password", "forgot password", "remember me",
    "войти", "пароль", "логин", "запомнить меня",
]
CRYPTO_PHRASES = ["seed phrase", "mnemonic", "private key", "wallet connect",
                  "connect wallet", "metamask", "trust wallet"]


def _domain_of(value: str, base_host: str) -> str:
    """Домен из src/href (абсолютный или относительный)."""
    v = value.strip()
    if not v or v.startswith(("#", "javascript:", "mailto:", "data:", "tel:")):
        return ""
    if v.startswith("//"):
        v = "http:" + v
    if "://" not in v:
        return base_host  # относительная ссылка — тот же хост
    try:
        return urlsplit(v).netloc.lower().partition(":")[0]
    except Exception:  # noqa: BLE001
        return ""


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_content_features(url: str, html: str) -> Dict[str, float]:
    """Признаки одной страницы. html может быть пустым/None."""
    f: Dict[str, float] = {}
    html = html or ""
    raw_len = len(html)
    f["c_len"] = float(raw_len)
    f["c_len_log"] = math.log1p(raw_len)
    f["c_empty"] = int(raw_len < 200)
    f["c_very_large"] = int(raw_len > MAX_HTML)

    u = normalize_url(url)
    host = urlsplit(u).netloc.lower().partition(":")[0].strip(".")
    _, sld, _ = split_host(host)
    doc = html[:MAX_HTML] if raw_len > MAX_HTML else html

    # ---------------- разбор тегов ----------------
    n_form = n_input = n_password = n_hidden = n_iframe = n_script = n_img = 0
    n_link = n_meta = n_button = n_select = n_textarea = n_a = 0
    n_event_handlers = 0
    external_domains = set()
    form_actions: List[str] = []
    iframe_domains: List[str] = []
    script_domains: List[str] = []
    favicon_domains: List[str] = []

    for m in TAG_RE.finditer(doc):
        tag = m.group(1).lower()
        attrs_raw = m.group(2) or ""
        if tag == "form":
            n_form += 1
            am = ATTR_RE.search(attrs_raw)
            for a in ATTR_RE.finditer(attrs_raw):
                if a.group(1).lower() == "action":
                    form_actions.append(a.group(2) or a.group(3) or a.group(4) or "")
        elif tag == "input":
            n_input += 1
            low = attrs_raw.lower()
            if "password" in low:
                n_password += 1
            if "hidden" in low:
                n_hidden += 1
        elif tag == "iframe":
            n_iframe += 1
            for a in ATTR_RE.finditer(attrs_raw):
                if a.group(1).lower() == "src":
                    iframe_domains.append(_domain_of(a.group(2) or a.group(3) or a.group(4) or "", host))
        elif tag == "script":
            n_script += 1
            for a in ATTR_RE.finditer(attrs_raw):
                if a.group(1).lower() == "src":
                    script_domains.append(_domain_of(a.group(2) or a.group(3) or a.group(4) or "", host))
        elif tag == "img":
            n_img += 1
            for a in ATTR_RE.finditer(attrs_raw):
                if a.group(1).lower() == "src":
                    d = _domain_of(a.group(2) or a.group(3) or a.group(4) or "", host)
                    if d:
                        external_domains.add(d)
        elif tag == "link":
            n_link += 1
            rel = ""
            href = ""
            for a in ATTR_RE.finditer(attrs_raw):
                k = a.group(1).lower()
                if k == "rel":
                    rel = (a.group(2) or a.group(3) or a.group(4) or "").lower()
                elif k == "href":
                    href = a.group(2) or a.group(3) or a.group(4) or ""
            if "icon" in rel:
                favicon_domains.append(_domain_of(href, host))
            d = _domain_of(href, host)
            if d:
                external_domains.add(d)
        elif tag == "a":
            n_a += 1
            for a in ATTR_RE.finditer(attrs_raw):
                if a.group(1).lower() == "href":
                    d = _domain_of(a.group(2) or a.group(3) or a.group(4) or "", host)
                    if d:
                        external_domains.add(d)
        elif tag == "meta":
            n_meta += 1
        elif tag == "button":
            n_button += 1
        elif tag == "select":
            n_select += 1
        elif tag == "textarea":
            n_textarea += 1
        if re.search(r"\bon[a-z]{3,}\s*=", attrs_raw, re.I):
            n_event_handlers += 1

    n_input_all = max(n_input, len(INPUT_RE.findall(doc)))
    f["c_n_form"] = n_form
    f["c_n_input"] = n_input_all
    f["c_n_password"] = n_password
    f["c_n_hidden"] = n_hidden
    f["c_n_iframe"] = n_iframe
    f["c_n_script"] = n_script
    f["c_n_img"] = n_img
    f["c_n_link"] = n_link
    f["c_n_meta"] = n_meta
    f["c_n_button"] = n_button
    f["c_n_select"] = n_select
    f["c_n_textarea"] = n_textarea
    f["c_n_a"] = n_a
    f["c_n_event_handlers"] = n_event_handlers
    f["c_has_password"] = int(n_password > 0)
    f["c_has_form"] = int(n_form > 0)
    f["c_password_no_form"] = int(n_password > 0 and n_form == 0)
    f["c_inputs_per_form"] = n_input_all / max(n_form, 1)

    # внешние домены
    ext = {d for d in external_domains if d and d != host and not d.endswith("." + host)}
    f["c_n_external_domains"] = len(ext)
    f["c_n_external_scripts"] = len([d for d in script_domains if d and d != host])
    f["c_n_external_iframes"] = len([d for d in iframe_domains if d and d != host])
    f["c_has_external_iframe"] = int(f["c_n_external_iframes"] > 0)
    f["c_favicon_external"] = int(any(d and d != host for d in favicon_domains))
    f["c_favicon_domain"] = favicon_domains[0] if favicon_domains else ""
    f["c_script_external_ratio"] = f["c_n_external_scripts"] / max(n_script, 1)

    # формы: куда уходят данные
    fa_ext = [a for a in form_actions if a and _domain_of(a, host) not in ("", host)]
    fa_http = [a for a in form_actions if a.lower().startswith("http://")]
    fa_php = [a for a in form_actions if ".php" in a.lower()]
    fa_mailto = [a for a in form_actions if a.lower().startswith("mailto:")]
    f["c_form_action_external"] = int(len(fa_ext) > 0)
    f["c_form_action_http"] = int(len(fa_http) > 0)
    f["c_form_action_php"] = int(len(fa_php) > 0)
    f["c_form_action_mailto"] = int(len(fa_mailto) > 0)
    f["c_form_action_empty"] = int(any(not a.strip() for a in form_actions))

    # ---------------- скрипты, редиректы, обфускация ----------------
    low_doc = doc.lower()
    f["c_meta_refresh"] = int(bool(META_REFRESH_RE.search(doc)))
    mr = META_REFRESH_URL_RE.search(doc)
    mr_dom = _domain_of(mr.group(1), host) if mr else ""
    f["c_meta_refresh_external"] = int(bool(mr_dom) and mr_dom != host)
    f["c_js_redirect"] = int(bool(JS_REDIRECT_RE.search(doc)))
    f["c_eval"] = int(bool(EVAL_RE.search(doc)))
    f["c_hex_escapes"] = len(HEXESC_RE.findall(doc))
    f["c_obfuscated_js"] = int(bool(OBFUSCATED_JS_RE.search(doc)))
    f["c_base64_blobs"] = len(BASE64_RE.findall(doc))
    f["c_hidden_style"] = int(("display:none" in low_doc.replace(" ", ""))
                              or ("visibility:hidden" in low_doc.replace(" ", "")))
    f["c_position_absolute"] = low_doc.count("position:absolute")
    f["c_z_index"] = low_doc.count("z-index")
    f["c_has_noscript"] = int("<noscript" in low_doc)

    # ---------------- текст страницы ----------------
    text = SCRIPT_STYLE_RE.sub(" ", doc)
    text = TEXT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    low_text = text.lower()
    f["c_text_len"] = len(text)
    f["c_text_words"] = float(len(low_text.split()))
    f["c_entropy_text"] = _entropy(text[:20000])
    f["c_cyrillic_ratio"] = len(CYRILLIC_RE.findall(text)) / max(len(text), 1)
    f["c_non_ascii_ratio"] = len(NON_ASCII_RE.findall(text)) / max(len(text), 1)
    f["c_digit_ratio_text"] = sum(ch.isdigit() for ch in text[:20000]) / max(min(len(text), 20000), 1)
    f["c_text_html_ratio"] = len(text) / max(len(doc), 1)

    # бренд в тексте страницы против домена
    brands_in_text = BRAND_RE.findall(low_text)
    brand_counter = Counter(b.lower() for b in brands_in_text)
    f["c_n_brand_mentions"] = len(brands_in_text)
    f["c_n_distinct_brands"] = len(brand_counter)
    f["c_max_brand_mentions"] = max(brand_counter.values()) if brand_counter else 0
    top_brand = brand_counter.most_common(1)[0][0] if brand_counter else ""
    f["c_top_brand"] = top_brand
    sld_norm = homoglyph_normalize(sld) if sld else ""
    brand_owned = bool(sld_norm) and any(
        sld_norm == b or sld_norm.startswith(b) or sld_norm.endswith(b) for b in CONTENT_BRANDS
    )
    f["c_brand_domain_mismatch"] = int(bool(top_brand) and not brand_owned)
    f["c_domain_brand_owned"] = int(brand_owned)
    # совпадение бренда в title с доменом
    t = re.search(r"<title[^>]*>(.*?)</title>", doc, re.I | re.S)
    title = WS_RE.sub(" ", TEXT_RE.sub(" ", t.group(1))).strip() if t else ""
    f["c_title_len"] = len(title)
    f["c_title_empty"] = int(len(title) == 0)
    title_brands = BRAND_RE.findall(title.lower())
    f["c_title_brand_mismatch"] = int(bool(title_brands) and not brand_owned)
    f["c_title_has_login"] = int(any(h in title.lower() for h in LOGIN_HINTS))
    f["c_title_equals_host"] = int(title.lower().strip() in (host, sld))

    # фразы социальной инженерии
    n_ph = sum(low_text.count(p) for p in PHISH_PHRASES)
    f["c_n_phish_phrases"] = n_ph
    f["c_has_phish_phrase"] = int(n_ph > 0)
    f["c_n_parked_phrases"] = sum(low_text.count(p) for p in PARKED_PHRASES)
    f["c_parked"] = int(f["c_n_parked_phrases"] > 0)
    f["c_n_login_hints"] = sum(low_text.count(p) for p in LOGIN_HINTS)
    f["c_n_crypto_phrases"] = sum(low_text.count(p) for p in CRYPTO_PHRASES)
    f["c_n_phish_words"] = sum(low_text.count(w) for w in PHISH_WORDS)
    f["c_phish_words_per_1k"] = f["c_n_phish_words"] / max(f["c_text_words"] / 1000.0, 1e-6)

    # комбинированные (наиболее «фишинговые») шаблоны
    f["c_login_form"] = int(n_password > 0 and (f["c_n_login_hints"] > 0 or f["c_title_has_login"]))
    f["c_clone_signature"] = int(
        (f["c_brand_domain_mismatch"] or f["c_title_brand_mismatch"])
        and (n_password > 0 or f["c_n_login_hints"] > 0)
    )
    f["c_steal_form"] = int(f["c_form_action_external"] and n_password > 0)
    f["c_redirect_chain"] = int(f["c_meta_refresh"] + f["c_js_redirect"] + f["c_has_external_iframe"])
    f["c_minimal_page"] = int(raw_len < 5000 and n_input_all > 0)
    f["c_suspicious_combo"] = int(sum([
        f["c_has_password"], f["c_n_external_scripts"] > 0, f["c_meta_refresh"],
        f["c_js_redirect"], f["c_eval"], f["c_base64_blobs"] > 0,
        f["c_brand_domain_mismatch"], f["c_has_phish_phrase"],
    ]) >= 4)
    return f


def _worker(args: Tuple[List[str], List[str]]) -> List[Dict[str, float]]:
    urls, htmls = args
    out = []
    for u, h in zip(urls, htmls):
        if not isinstance(h, str):
            h = "" if h is None or (isinstance(h, float) and np.isnan(h)) else str(h)
        try:
            out.append(extract_content_features(u, h[:MAX_HTML_HARD]))
        except Exception as exc:  # noqa: BLE001
            out.append({"c_error": 1.0, "c_len": float(len(h))})
    return out


def feature_columns() -> List[str]:
    """Полный список числовых колонок (строковые признаки отбрасываются)."""
    sample = extract_content_features("http://example.com/", "<html><title>x</title><body>ok</body></html>")
    return [k for k, v in sample.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]


def build_content_features(csv_path: str, out_path: str, workers: int = 8,
                           chunksize: int = 2000, limit: int | None = None,
                           url_col: str = "url", content_col: str = "content") -> pd.DataFrame:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames: List[pd.DataFrame] = []
    processed = 0
    reader = pd.read_csv(csv_path, chunksize=chunksize, on_bad_lines="skip",
                         engine="c", dtype=str, keep_default_na=False)
    with Pool(workers) as pool:
        from tqdm import tqdm
        for chunk in reader:
            if url_col not in chunk.columns:
                url_col = chunk.columns[0]
            if content_col not in chunk.columns:
                content_col = chunk.columns[-1]
            urls = chunk[url_col].astype(str).tolist()
            htmls = chunk[content_col].astype(str).tolist()
            n = len(urls)
            parts = [(urls[i:i + max(n // workers + 1, 1)], htmls[i:i + max(n // workers + 1, 1)])
                     for i in range(0, n, max(n // workers + 1, 1))]
            feats: List[Dict[str, float]] = []
            for res in pool.imap(_worker, parts):
                feats.extend(res)
            df = pd.DataFrame(feats)
            df["content_url"] = urls
            df["content_missing"] = [int(len(h) < 200) for h in htmls]
            frames.append(df)
            processed += n
            print(f"  обработано {processed} строк", flush=True)
            if limit and processed >= limit:
                break
    result = pd.concat(frames, ignore_index=True)
    for c in result.columns:
        if c not in ("content_url", "c_top_brand", "c_favicon_domain"):
            result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0.0)
    result.to_pickle(out_path)
    print(f"[ok] {out_path}: {result.shape}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Признаки HTML-контента страниц")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    ap.add_argument("--chunksize", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None, help="обработать только N строк (тест)")
    args = ap.parse_args()
    build_content_features(args.csv, args.out, workers=args.workers,
                           chunksize=args.chunksize, limit=args.limit)


if __name__ == "__main__":
    main()
