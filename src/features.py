# -*- coding: utf-8 -*-
"""
features.py — извлечение признаков из URL для задачи обнаружения фишинга.

Реализовано три группы признаков:

1. Лексико-структурные (handcrafted): длина, количество точек/дефисов/цифр,
   энтропия, IP-адрес вместо домена, порт, "@" в URL, punycode, процентное
   кодирование, расширения файлов и т.д.
2. Признаки "социальной инженерии": слова-приманки (login, verify, bonus,
   prize, wallet...), брендовые слова вне домена (мимикрия под безопасный сайт).
3. Признаки тайпсквоттинга: расстояние Левенштейна/Дамерау до списка
   популярных брендов, гомоглифные подмены (paypa1 -> paypal, rnicrosoft -> microsoft),
   вставка дефисов, бренд в поддомене или в пути.

Все признаки считаются без обращения к сети (offline), что позволяет
обрабатывать десятки тысяч URL за секунды.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple
from urllib.parse import unquote, urlsplit

import numpy as np

# --------------------------------------------------------------------------
# Справочники (можно расширять; при желании подгружаются из data/*.txt)
# --------------------------------------------------------------------------

# Популярные бренды: используются для поиска тайпсквоттинга и мимикрии.
BRANDS: List[str] = [
    "google", "youtube", "facebook", "amazon", "wikipedia", "twitter", "instagram",
    "linkedin", "netflix", "microsoft", "apple", "icloud", "office", "outlook",
    "paypal", "visa", "mastercard", "mir", "sberbank", "sber", "alfabank", "alfa",
    "tinkoff", "tbank", "vtb", "gazprombank", "raiffeisen", "otkritie", "psbank",
    "yandex", "mail", "vk", "ok", "telegram", "whatsapp", "discord", "tiktok",
    "steam", "roblox", "epicgames", "playstation", "xbox", "nintendo", "origin",
    "binance", "coinbase", "blockchain", "metamask", "trustwallet", "kraken",
    "aliexpress", "alibaba", "ebay", "wildberries", "ozon", "avito", "dns", "mvideo",
    "gosuslugi", "nalog", "pfrf", "mos", "mosreg", "rt", "pochta", "russianpost",
    "cdek", "boxberry", "delivery", "dhl", "fedex", "ups", "dpd",
    "dropbox", "adobe", "docusign", "zoom", "slack", "github", "gitlab", "bitbucket",
    "yahoo", "hotmail", "gmail", "protonmail", "rambler", "bank", "onlinebank",
    "chase", "wellsfargo", "citibank", "hsbc", "barclays", "santander", "revolut",
    "uber", "airbnb", "booking", "tripadvisor", "expedia", "lufthansa", "aeroflot",
    "sbermarket", "yandexmarket", "magnit", "pyaterochka", "lenta", "auchan",
]

# Домены-«однодневки» / бесплатные хостинги: часто используются для фишинга.
FREE_HOSTS = [
    "000webhostapp", "weebly", "wixsite", "blogspot", "wordpress", "github.io",
    "gitlab.io", "netlify", "vercel", "glitch", "repl.co", "herokuapp",
    "web.app", "firebaseapp", "pages.dev", "workers.dev", "myshopify",
    "square.site", "webnode", "jimdofree", "ucoz", "narod", "tilda.ws",
    "beget.tech", "hostinger", "byethost", "infinityfree", "altervista",
]

# Сервисы сокращения ссылок (используются для сокрытия реального адреса).
SHORTENERS = [
    "bit.ly", "goo.gl", "tinyurl", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly",
    "bitly", "cutt.ly", "rb.gy", "shorturl", "rebrand.ly", "tiny.cc", "clck.ru",
    "vk.cc", "goo.su", "u.to", "qps.ru", "burli", "shorte.st", "ouo.io",
]

# Домены верхнего уровня с высокой долей злоупотреблений.
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "click", "link",
    "live", "online", "site", "website", "space", "fun", "icu", "cyber", "rest",
    "bar", "beauty", "casa", "buzz", "monster", "quest", "sbs", "cam", "surf",
    "shop", "store", "pw", "cc", "su", "ws", "za.com", "country", "stream",
    "download", "racing", "loan", "win", "bid", "trade", "date", "review",
}

# Слова-приманки социальной инженерии.
PHISH_WORDS = [
    "login", "log-in", "signin", "sign-in", "logon", "verify", "verification",
    "confirm", "confirmation", "account", "update", "secure", "security",
    "password", "passwd", "pwd", "credential", "auth", "authenticate", "token",
    "session", "unlock", "blocked", "suspended", "limited", "recovery", "recover",
    "reset", "support", "helpdesk", "service", "billing", "invoice", "payment",
    "pay", "wallet", "refund", "bonus", "gift", "prize", "winner", "win", "free",
    "promo", "cashback", "lucky", "claim", "reward", "survey", "giveaway",
    "delivery", "parcel", "tracking", "order", "checkout", "subscribe",
    "webscr", "cmd", "dispatch", "administrator", "webmaster", "root",
]

# Двухчастные публичные суффиксы (упрощённый публичный суффикс-лист).
MULTI_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "net.uk", "sch.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au", "id.au",
    "co.nz", "net.nz", "org.nz", "govt.nz",
    "co.in", "net.in", "org.in", "gen.in", "firm.in", "gov.in", "ac.in",
    "com.br", "net.br", "org.br", "gov.br", "edu.br",
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn", "ac.cn",
    "co.jp", "ne.jp", "or.jp", "go.jp", "ac.jp", "ad.jp",
    "com.tr", "net.tr", "org.tr", "gov.tr", "edu.tr",
    "co.za", "org.za", "net.za", "gov.za",
    "com.mx", "com.ar", "com.co", "com.pe", "com.ve", "com.ua", "net.ua",
    "org.ua", "gov.ua", "kiev.ua", "com.ru", "net.ru", "org.ru", "msk.ru",
    "spb.ru", "co.kr", "or.kr", "go.kr", "com.sg", "com.hk", "com.tw",
    "com.my", "com.ph", "com.vn", "com.pk", "com.eg", "com.sa", "com.ng",
    "co.il", "co.th", "co.id", "com.es", "com.pl", "com.ro", "com.gr",
    "co.at", "or.at", "co.hu", "com.pt", "com.se", "com.no", "com.fi",
    "com.dk", "com.cz", "com.be", "com.ch", "com.it", "com.fr", "com.de",
    "co.ke", "co.tz", "co.ug", "com.gh", "co.bw", "co.mz",
}

IPV4_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
HEX_RE = re.compile(r"%[0-9a-fA-F]{2}")
PUNY_RE = re.compile(r"xn--", re.I)
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")

# Гомоглифные замены: приводим «визуальные подделки» к буквам латиницы.
HOMOGLYPHS = {
    "0": "o", "1": "l", "2": "z", "3": "e", "4": "a", "5": "s", "6": "g",
    "7": "t", "8": "b", "9": "g", "@": "a", "$": "s", "!": "i", "|": "l",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "к": "k", "в": "b", "н": "h", "м": "m", "т": "t", "і": "i", "ѕ": "s",
}
# Пары букв, визуально образующие одну букву.
DIGRAPH_FIXES = [("rn", "m"), ("vv", "w"), ("cl", "d"), ("nn", "m")]


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------

def entropy(text: str) -> float:
    """Энтропия Шеннона строки (мера «случайности»)."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _levenshtein_bounded(a: str, b: str, max_dist: int = 2) -> int:
    """Расстояние Левенштейна с ранним выходом (более max_dist -> max_dist+1)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        best = cur[0]
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            best = min(best, cur[j])
        if best > max_dist:
            return max_dist + 1
        prev = cur
    return prev[lb]


def _levenshtein_factory():
    """Быстрая реализация, если доступен rapidfuzz; иначе чистый Python."""
    try:  # pragma: no cover
        from rapidfuzz.distance import Levenshtein as _RF

        def _dist(a: str, b: str, max_dist: int = 2) -> int:
            if a == b:
                return 0
            if abs(len(a) - len(b)) > max_dist:
                return max_dist + 1
            return int(_RF.distance(a, b, score_cutoff=max_dist + 1))

        return _dist
    except Exception:
        return _levenshtein_bounded


levenshtein = _levenshtein_factory()


def homoglyph_normalize(text: str) -> str:
    """Сводит гомоглифы к латинским буквам: paypa1 -> paypal, rnicro -> micro."""
    if not text:
        return ""
    out = []
    for ch in text.lower():
        out.append(HOMOGLYPHS.get(ch, ch))
    s = "".join(out)
    for src, dst in DIGRAPH_FIXES:
        s = s.replace(src, dst)
    return s


def split_host(host: str) -> Tuple[str, str, str]:
    """
    Возвращает (subdomain, domain, suffix) — упрощённый аналог tldextract.
    Пример: login.paypal.com.ru -> ("login", "paypal", "com.ru")
    """
    host = (host or "").strip(".").lower()
    if not host:
        return "", "", ""
    parts = host.split(".")
    if len(parts) == 1:
        return "", parts[0], ""
    last2 = ".".join(parts[-2:])
    if last2 in MULTI_SUFFIXES and len(parts) >= 3:
        suffix = last2
        domain = parts[-3]
        sub = ".".join(parts[:-3])
        return sub, domain, suffix
    suffix = parts[-1]
    domain = parts[-2]
    sub = ".".join(parts[:-2])
    return sub, domain, suffix


def normalize_url(raw: str) -> str:
    """Добавляет схему, если её нет (urlparse иначе считает домен путём)."""
    u = str(raw).strip().strip('"').strip("'")
    u = u.replace("\u200b", "").replace("\ufeff", "")
    if not u:
        return ""
    low = u.lower()
    if low.startswith("//"):
        return "http:" + u
    if not re.match(r"^[a-z][a-z0-9+.\-]*://", low):
        # Нет схемы: "example.com/path", "192.168.0.1:8080/x", "mailto:..." и пр.
        if low.startswith("mailto:") or low.startswith("javascript:"):
            return "http://" + u.split(":", 1)[1]
        return "http://" + u
    return u


_NUM_TOKENS = re.compile(r"[a-z0-9]+")
_WORD_TOKENS = re.compile(r"[a-z]{3,}")


def _tokens(text: str) -> List[str]:
    return _NUM_TOKENS.findall(text.lower())


# --------------------------------------------------------------------------
# Главная функция извлечения признаков
# --------------------------------------------------------------------------

def extract_features(raw_url: str) -> Dict[str, float]:
    """Возвращает словарь признаков для одного URL."""
    f: Dict[str, float] = {}
    u = normalize_url(raw_url)
    low = u.lower()

    parts = urlsplit(u)
    scheme = parts.scheme.lower()
    host_raw = parts.netloc
    # userinfo (user:pass@host) — классический приём обмана
    userinfo = ""
    host = host_raw
    if "@" in host_raw:
        userinfo, host = host_raw.rsplit("@", 1)
    host_port = host
    host_only, _, port = host.partition(":")
    host_only = host_only.lower().strip(".")
    path = parts.path or ""
    query = parts.query or ""
    fragment = parts.fragment or ""

    sub, domain, suffix = split_host(host_only)
    sld = domain  # second-level domain = «имя» домена
    decoded_path = unquote(path)
    decoded_query = unquote(query)
    full_text = f"{host_only}{path}{query}"

    # ---------------- 1. Лексико-структурные признаки ----------------
    f["url_len"] = len(u)
    f["host_len"] = len(host_only)
    f["path_len"] = len(path)
    f["query_len"] = len(query)
    f["fragment_len"] = len(fragment)
    f["domain_len"] = len(sld)
    f["suffix_len"] = len(suffix)
    f["subdomain_len"] = len(sub)

    f["n_dots"] = u.count(".")
    f["n_dots_host"] = host_only.count(".")
    f["n_subdomains"] = len([p for p in sub.split(".") if p]) if sub else 0
    f["n_hyphens"] = u.count("-")
    f["n_hyphens_host"] = host_only.count("-")
    f["n_hyphens_sld"] = sld.count("-")
    f["n_underscores"] = u.count("_")
    f["n_slashes"] = u.count("/")
    f["n_double_slash"] = max(low.count("//") - (1 if "://" in low else 0), 0)
    f["n_digits"] = sum(ch.isdigit() for ch in u)
    f["n_digits_host"] = sum(ch.isdigit() for ch in host_only)
    f["n_letters_host"] = sum(ch.isalpha() for ch in host_only)
    f["digits_ratio_url"] = f["n_digits"] / max(len(u), 1)
    f["digits_ratio_host"] = f["n_digits_host"] / max(len(host_only), 1)
    f["n_upper"] = sum(ch.isupper() for ch in u)
    f["upper_ratio"] = f["n_upper"] / max(len(u), 1)
    f["n_special"] = sum(ch in "?=&%$#@!*'\"();:+,<>[]{}|\\^~`" for ch in u)
    f["n_ampersand"] = u.count("&")
    f["n_equal"] = u.count("=")
    f["n_percent"] = u.count("%")
    f["n_question"] = u.count("?")
    f["n_at"] = u.count("@")
    f["n_comma"] = u.count(",")
    f["n_semicolon"] = u.count(";")
    f["n_tilde"] = u.count("~")
    f["n_plus"] = u.count("+")
    f["n_space"] = u.count(" ") + u.count("%20") + u.count("+")

    f["has_scheme"] = int(bool(parts.scheme))
    f["is_https"] = int(scheme == "https")
    f["is_http"] = int(scheme == "http")
    f["is_other_scheme"] = int(scheme not in ("http", "https"))
    f["has_port"] = int(bool(port))
    f["port_is_standard"] = int(port in ("", "80", "443"))
    f["port_is_suspicious"] = int(bool(port) and port not in ("80", "443", "8080"))
    f["has_userinfo"] = int(bool(userinfo))
    f["has_www"] = int(host_only.startswith("www."))
    f["has_fragment"] = int(bool(fragment))
    f["has_query"] = int(bool(query))
    f["n_query_params"] = len([p for p in query.split("&") if p])

    f["is_ip_host"] = int(bool(IPV4_RE.match(host_only)))
    f["is_ip_in_url"] = int(bool(IPV4_RE.search(u)))
    f["is_punycode"] = int(bool(PUNY_RE.search(host_only)))
    f["has_non_ascii"] = int(bool(NON_ASCII_RE.search(raw_url if isinstance(raw_url, str) else u)))
    f["n_non_ascii"] = len(NON_ASCII_RE.findall(u))
    f["has_hex_encoding"] = int(bool(HEX_RE.search(path + query)))
    f["n_hex_encoding"] = len(HEX_RE.findall(path + query))
    f["has_unescaped_space"] = int(" " in u)

    f["entropy_url"] = entropy(u)
    f["entropy_host"] = entropy(host_only)
    f["entropy_domain"] = entropy(sld)
    f["entropy_path"] = entropy(path)
    f["entropy_query"] = entropy(query)
    f["entropy_full"] = entropy(full_text)

    f["path_depth"] = len([p for p in path.split("/") if p])
    f["path_max_token_len"] = max((len(t) for t in _tokens(path)), default=0)
    f["path_avg_token_len"] = float(np.mean([len(t) for t in _tokens(path)])) if _tokens(path) else 0.0
    f["host_max_token_len"] = max((len(t) for t in host_only.split(".")), default=0)
    f["longest_run_digits"] = max((len(m) for m in re.findall(r"\d+", u)), default=0)
    f["longest_run_letters"] = max((len(m) for m in re.findall(r"[a-zA-Z]+", u)), default=0)
    f["digit_letter_transitions"] = len(re.findall(r"(?<=[a-z])\d|\d(?=[a-z])", low))

    ext = ""
    if "." in path.rsplit("/", 1)[-1]:
        ext = path.rsplit("/", 1)[-1].rsplit(".", 1)[-1].lower()
    f["path_has_extension"] = int(bool(ext) and len(ext) <= 5)
    f["ext_is_php"] = int(ext in ("php", "php3", "php5", "phtml", "asp", "aspx", "jsp", "cgi"))
    f["ext_is_html"] = int(ext in ("html", "htm", "shtml"))
    f["ext_is_exec"] = int(ext in ("exe", "apk", "msi", "zip", "rar", "scr", "js", "jar", "bat", "cmd"))
    f["double_extension"] = int(bool(re.search(r"\.(jpg|png|gif|doc|pdf|txt)\.(php|html?|exe|js)$", path, re.I)))
    f["path_ends_with_slash"] = int(path.endswith("/"))

    f["tld"] = suffix
    f["suspicious_tld"] = int(suffix in SUSPICIOUS_TLDS or suffix.split(".")[-1] in SUSPICIOUS_TLDS)
    f["is_free_host"] = int(any(h in host_only for h in FREE_HOSTS))
    f["is_shortener"] = int(any(s in host_only for s in SHORTENERS))
    f["sld_has_digit"] = int(any(ch.isdigit() for ch in sld))
    f["sld_has_hyphen"] = int("-" in sld)
    f["sld_is_short"] = int(0 < len(sld) <= 5)
    f["host_is_long"] = int(len(host_only) > 30)
    f["url_is_long"] = int(len(u) > 75)
    f["url_is_very_long"] = int(len(u) > 120)
    f["many_subdomains"] = int(f["n_subdomains"] >= 3)
    f["many_hyphens_host"] = int(host_only.count("-") >= 2)
    f["suspicious_combo"] = int(
        (f["n_subdomains"] >= 2) + (f["n_hyphens_host"] >= 1) + f["suspicious_tld"]
        + f["has_userinfo"] + f["is_ip_host"] + (f["url_len"] > 75) >= 3
    )

    # ---------------- 2. Слова-приманки и бренды ----------------
    words = set(_WORD_TOKENS.findall(full_text))
    words.update(_WORD_TOKENS.findall(host_only.replace("-", " ").replace(".", " ")))
    hits = [w for w in PHISH_WORDS if w in full_text]
    f["n_phish_words"] = len(hits)
    f["phish_word_ratio"] = len(hits) / max(len(words), 1)
    f["has_login_word"] = int(any(w in full_text for w in ("login", "signin", "logon", "log-in", "sign-in")))
    f["has_secure_word"] = int(any(w in full_text for w in ("secure", "security", "ssl", "safe")))
    f["has_verify_word"] = int(any(w in full_text for w in ("verify", "confirm", "validation", "update")))
    f["has_account_word"] = int(any(w in full_text for w in ("account", "profile", "user", "customer")))
    f["has_money_word"] = int(any(w in full_text for w in ("pay", "bank", "card", "wallet", "billing", "invoice", "refund")))
    f["has_prize_word"] = int(any(w in full_text for w in ("bonus", "prize", "gift", "winner", "free", "promo", "cashback", "lucky", "claim", "reward")))
    f["has_delivery_word"] = int(any(w in full_text for w in ("delivery", "parcel", "tracking", "order", "shipment")))

    brand_hits = [b for b in BRANDS if b in full_text]
    f["n_brand_words"] = len(brand_hits)
    # «Честный» домен бренда: SLD в точности равен названию бренда.
    f["brand_in_domain"] = int(sld in BRANDS)
    # Бренд «вклеен» в более длинный SLD: sberbank-online, paypal-secure, sberbank24.
    f["brand_embedded_in_sld"] = int(
        len(sld) > 3 and sld not in BRANDS and any(b in sld and len(b) >= 3 for b in BRANDS)
    )
    f["brand_in_subdomain"] = int(any(b in sub for b in BRANDS)) if sub else 0
    f["brand_in_path"] = int(any(b in decoded_path.lower() for b in BRANDS))
    f["brand_in_query"] = int(any(b in decoded_query.lower() for b in BRANDS))
    f["brand_position"] = int(
        2 if f["brand_in_subdomain"] else 1 if f["brand_embedded_in_sld"] else 0
    )
    # Мимикрия: бренд присутствует в URL, но домен НЕ принадлежит бренду.
    f["brand_mimicry"] = int(
        (not f["brand_in_domain"])
        and (f["brand_in_subdomain"] or f["brand_in_path"] or f["brand_in_query"]
             or f["brand_embedded_in_sld"])
    )
    f["brand_plus_phish_word"] = int(f["n_brand_words"] > 0 and f["n_phish_words"] > 0)

    # ---------------- 3. Тайпсквоттинг ----------------
    # Для IP-адресов и очень коротких SLD сравнение с брендами бессмысленно.
    is_ip = bool(IPV4_RE.match(host_only))
    if sld and not is_ip and len(sld) >= 4:
        norm = homoglyph_normalize(sld)
        best_dist, best_brand, homoglyph_hit, hyphen_hit = 99, "", 0, 0
        for b in BRANDS:
            # Быстрый отсев: сравниваем только «похожие по длине» бренды
            if abs(len(sld) - len(b)) > 3:
                continue
            d = levenshtein(sld, b, 2)
            if d < best_dist:
                best_dist, best_brand = d, b
            if norm == b and sld != b:
                homoglyph_hit = 1
            if sld.replace("-", "") == b and sld != b:
                hyphen_hit = 1
            # бренд как часть SLD с добавкой: paypal-secure, secure-paypal
            if sld != b and len(b) >= 4 and b in sld:
                hyphen_hit = max(hyphen_hit, 1)
        f["min_brand_distance"] = float(best_dist if best_dist <= 2 else 3)
        f["is_typosquatting"] = int(1 <= best_dist <= 2)
        f["is_homoglyph_attack"] = homoglyph_hit
        f["is_hyphen_brand_attack"] = int(bool(hyphen_hit))
        f["sld_keyboard_distance"] = float(min(best_dist, 3))
    else:
        f["min_brand_distance"] = 3.0
        f["is_typosquatting"] = 0
        f["is_homoglyph_attack"] = 0
        f["is_hyphen_brand_attack"] = 0
        f["sld_keyboard_distance"] = 3.0

    f["attack_typosquatting"] = int(
        f["is_typosquatting"] or f["is_homoglyph_attack"] or f["is_hyphen_brand_attack"]
    )
    f["attack_mimicry"] = int(f["brand_mimicry"])
    f["attack_keywords"] = int(f["n_phish_words"] >= 2)
    f["n_attack_techniques"] = f["attack_typosquatting"] + f["attack_mimicry"] + f["attack_keywords"]

    # ---------------- 4. Прочее ----------------
    f["host_letter_entropy_ratio"] = f["entropy_host"] / max(f["host_len"], 1)
    f["path_letter_entropy_ratio"] = f["entropy_path"] / max(f["path_len"], 1)
    f["vowels_ratio_host"] = sum(ch in "aeiouy" for ch in host_only) / max(len(host_only), 1)
    f["consecutive_consonants"] = float(
        max((len(m) for m in re.findall(r"[bcdfghjklmnpqrstvwxz]+", host_only)), default=0)
    )
    f["tld_is_country"] = int(len(suffix) == 2 and suffix.isalpha())
    f["tld_is_generic"] = int(suffix in ("com", "org", "net", "edu", "gov", "int", "mil"))

    return f


FEATURE_GROUPS = {
    "lexical": ["url_len", "host_len", "path_len", "n_dots", "n_hyphens", "n_digits"],
    "security": ["is_https", "has_userinfo", "is_ip_host", "suspicious_tld"],
    "social": ["n_phish_words", "brand_mimicry", "has_prize_word"],
    "typosquatting": ["is_typosquatting", "is_homoglyph_attack", "min_brand_distance"],
}


def build_feature_matrix(urls, verbose: bool = True):
    """Строит DataFrame признаков для последовательности URL."""
    import pandas as pd
    from tqdm import tqdm

    iterator = tqdm(urls, desc="features", ncols=80, disable=not verbose)
    rows = [extract_features(u) for u in iterator]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Текстовая модель: символьные n-граммы (используется как второй вид)
# --------------------------------------------------------------------------

def char_ngram_text(raw_url: str) -> str:
    """
    Нормализует URL для символьного TF-IDF:
    убираем схему, заменяем цифры на 'D', редкие символы на пробел-разделители.
    """
    u = normalize_url(raw_url).lower()
    u = re.sub(r"^[a-z]+://", "", u)
    u = re.sub(r"\d", "0", u)
    u = re.sub(r"[^a-z0-9./?=&\-_]", " ", u)
    u = re.sub(r"[./?=&\-_]", lambda m: " " + m.group(0) + " ", u)
    return re.sub(r"\s+", " ", u).strip()


def url_word_text(raw_url: str) -> str:
    """Текст для word-level TF-IDF: токены хоста и пути, разделённые дефисами/точками."""
    u = normalize_url(raw_url).lower()
    u = re.sub(r"^[a-z]+://", "", u)
    u = re.split(r"[?#]", u)[0]
    toks = re.split(r"[^a-z0-9]+", u)
    norm = []
    for t in toks:
        if not t:
            continue
        norm.append(homoglyph_normalize(t))
    return " ".join(norm + toks)


def find_typosquat_brand(host: str, max_dist: int = 2) -> Tuple[str, int]:
    """
    Для анализа техник атак: возвращает (бренд, расстояние), которому
    подражает домен. ('', 99) — если подражание не обнаружено.

    Пример: find_typosquat_brand("paypa1.com") -> ("paypal", 1)
            find_typosquat_brand("rnicrosoft-support.tk") -> ("microsoft", 1)
    """
    _, sld, _ = split_host(host)
    if not sld or IPV4_RE.match(host or "") or len(sld) < 4:
        return "", 99
    best, best_d = "", 99
    norm = homoglyph_normalize(sld)
    for b in BRANDS:
        if sld == b:  # домен самого бренда — это не тайпсквоттинг
            continue
        if abs(len(sld) - len(b)) > 3:
            continue
        # 0 после нормализации означает «визуально неотличим» (paypa1 -> paypal)
        d = min(levenshtein(sld, b, max_dist), levenshtein(norm, b, max_dist))
        if d < best_d:
            best, best_d = b, d
    return (best, best_d) if best_d <= max_dist else ("", 99)


def find_mimicked_brand(url: str) -> str:
    """
    Возвращает бренд, под который маскируется URL (мимикрия),
    или пустую строку. Используется только для отчётности/аналитики.
    """
    u = normalize_url(url)
    parts = urlsplit(u)
    host = parts.netloc.lower()
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    host = host.partition(":")[0].strip(".")
    sub, sld, _ = split_host(host)
    text = f"{sub} {parts.path} {parts.query}".lower()
    if sld in BRANDS:
        return ""
    for b in BRANDS:
        if len(b) >= 4 and b in text:
            return b
    for b in BRANDS:
        if len(b) >= 4 and b in sld:
            return b
    return ""
