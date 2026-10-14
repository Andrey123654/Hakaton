# -*- coding: utf-8 -*-
"""
rdap_features.py — признаки регистрации домена по RDAP.

Возраст домена — классический признак фишинга: мошеннические домены
регистрируют за дни или недели до атаки, тогда как легитимные существуют
годами. RDAP (Registration Data Access Protocol) — HTTP/JSON-замена WHOIS,
работает быстрее и стабильнее.

Считаются: возраст домена (дни с регистрации), срок до истечения, число
статусов EPP (clientHold, pendingDelete и т.п.), наличие регистратора,
число NS, а также отдельный признак «домен не найден в реестре» (404) —
это значит, что домен удалён или никогда не был зарегистрирован.

Адресация к реестрам берётся из bootstrap-файла IANA, поэтому покрываются
все зоны, поддерживающие RDAP (.com/.net/.org/.info/.xyz и др.).

Воспроизводимость: снимок сохраняется в файл и поставляется с кодом
(данные регистрации со временем меняются).

Запуск:
    python src/rdap_features.py --out cache/rdap_all.pkl --workers 20
    python src/rdap_features.py --sample 1500 --out /tmp/rdap_sample.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dns_features import host_of  # noqa: E402
from features import split_host  # noqa: E402

BOOTSTRAP_URL = "https://data.iana.org/rdap/dns.json"
CTX = ssl.create_default_context()
UA = {"User-Agent": "phishing-research-bot/1.0", "Accept": "application/rdap+json"}
SUSPICIOUS_STATUS = {"clienthold", "pendingdelete", "redemptionperiod",
                     "pendingdelete restore", "inactive"}


def load_bootstrap(cache_path: str = "cache/rdap_bootstrap.json") -> Dict[str, str]:
    """Карта «зона -> базовый URL RDAP» из реестра IANA."""
    data = None
    if os.path.exists(cache_path):
        try:
            data = json.load(open(cache_path, encoding="utf-8"))
        except Exception:
            data = None
    if data is None:
        try:
            req = urllib.request.Request(BOOTSTRAP_URL, headers={"User-Agent": "research-bot"})
            with urllib.request.urlopen(req, timeout=15, context=CTX) as r:
                data = json.loads(r.read().decode("utf-8"))
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            json.dump(data, open(cache_path, "w", encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] bootstrap IANA недоступен: {exc}")
            return {}
    mapping: Dict[str, str] = {}
    for services in data.get("services", []):
        if len(services) < 2:
            continue
        tlds, urls = services[0], services[1]
        base = next((u for u in urls if u.startswith("https")), urls[0] if urls else None)
        if not base:
            continue
        for t in tlds:
            mapping[t.lower()] = base.rstrip("/")
    return mapping


def registrable_domain(url_or_host: str) -> Tuple[str, str]:
    """(регистрируемый домен, зона)."""
    h = host_of(url_or_host) if "://" in url_or_host else url_or_host
    _, sld, suf = split_host(h)
    if suf and sld:
        return f"{sld}.{suf}", suf.split(".")[-1].lower()
    return h, ""


def rdap_record(domain: str, base: str, attempts: int = 2) -> Optional[dict]:
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(f"{base}/domain/{domain}", headers=UA)
            with urllib.request.urlopen(req, timeout=8, context=CTX) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"_not_found": True}
            if e.code in (429, 500, 502, 503) and attempt < attempts - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
        except Exception:
            if attempt < attempts - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            return None
    return None


EMPTY = {"rdap_ok": 0.0, "rdap_not_found": 0.0, "rdap_age_days": -1.0,
         "rdap_days_to_exp": -1.0, "rdap_n_status": -1.0, "rdap_susp_status": -1.0,
         "rdap_has_registrar": -1.0, "rdap_n_ns": -1.0, "rdap_age_log": -1.0}


def domain_features(domain: str, bootstrap: Dict[str, str],
                    attempts: int = 2) -> Dict[str, float]:
    _, tld = registrable_domain(domain)
    base = bootstrap.get(tld)
    if not base:
        return {**EMPTY, "rdap_no_service": 1.0}
    rec = rdap_record(domain, base, attempts=attempts)
    if rec is None:
        return dict(EMPTY)
    if rec.get("_not_found"):
        return {**EMPTY, "rdap_not_found": 1.0}
    f = dict(EMPTY)
    f["rdap_ok"] = 1.0
    now = datetime.now(timezone.utc)
    for ev in rec.get("events", []):
        action = str(ev.get("eventAction", "")).lower()
        try:
            dt = datetime.fromisoformat(str(ev.get("eventDate", "")).replace("Z", "+00:00"))
            if dt.tzinfo is None:          # часть реестров отдаёт дату без зоны
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if action == "registration":
            age = (now - dt).days
            f["rdap_age_days"] = float(age)
            f["rdap_age_log"] = float(np.log1p(max(age, 0)))
        elif action == "expiration":
            f["rdap_days_to_exp"] = float((dt - now).days)
    statuses = [str(s).lower() for s in rec.get("status", []) or []]
    f["rdap_n_status"] = float(len(statuses))
    f["rdap_susp_status"] = float(any(any(k in s for k in SUSPICIOUS_STATUS) for s in statuses))
    f["rdap_has_registrar"] = float(bool(rec.get("registrar")))
    f["rdap_n_ns"] = float(len(rec.get("nameservers", []) or []))
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description="Признаки регистрации домена (RDAP)")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default="cache/rdap_all.pkl")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--sample", type=int, default=0,
                    help="случайная сбалансированная выборка строк train для проверки")
    ap.add_argument("--retry-failed", action="store_true",
                    help="повторно опросить домены, для которых RDAP не ответил, "
                         "и обновить существующий снимок")
    args = ap.parse_args()

    bootstrap = load_bootstrap()
    print(f"[info] зон с RDAP: {len(bootstrap)}")

    if args.retry_failed:
        table = pd.read_pickle(args.out)
        failed = table[(table["rdap_ok"] == 0) & (table["rdap_not_found"] == 0)
                       & (table.get("rdap_no_service", 0) == 0)].index.tolist()
        print(f"[info] повторный опрос {len(failed)} доменов "
              f"({args.workers} потоков, 3 попытки)")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(lambda d: domain_features(d, bootstrap, attempts=3), failed))
        fixed = 0
        for dom, rec in zip(failed, rows):
            if rec.get("rdap_ok") == 1 or rec.get("rdap_not_found") == 1:
                table.loc[dom] = pd.Series({**{c: table.loc[dom, c] for c in table.columns},
                                            **rec})
                fixed += 1
            else:
                for k, v in rec.items():
                    table.loc[dom, k] = v
        table.to_pickle(args.out)
        print(f"[ok] восстановлено записей: {fixed} из {len(failed)} "
              f"за {time.time() - t0:.0f} с")
        print(f"[info] итог: RDAP-ответов {table['rdap_ok'].mean() * 100:.1f} %, "
              f"не найдено {table['rdap_not_found'].mean() * 100:.1f} %")
        return

    urls = pd.read_csv(args.train)["url"].astype(str)
    if args.sample:
        y = pd.read_csv(args.train)["result"].values
        rng = np.random.RandomState(0)
        idx = np.r_[rng.choice(np.where(y == 1)[0], args.sample // 2, replace=False),
                    rng.choice(np.where(y == 0)[0], args.sample // 2, replace=False)]
        urls = urls.iloc[idx]
    domains = sorted({registrable_domain(u)[0] for u in urls})
    print(f"[info] доменов для запроса: {len(domains)} ({args.workers} потоков)")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(lambda d: domain_features(d, bootstrap), domains))
    print(f"[info] опрошено за {time.time() - t0:.0f} с")
    table = pd.DataFrame(rows, index=domains)
    table.index.name = "domain"
    if "rdap_no_service" not in table.columns:
        table["rdap_no_service"] = 0.0
    table = table.fillna(0.0)
    table.to_pickle(args.out)
    print(f"[ok] {args.out}: {table.shape}; ответов RDAP {table['rdap_ok'].mean() * 100:.1f} %, "
          f"не найдено в реестре {table['rdap_not_found'].mean() * 100:.1f} %")


if __name__ == "__main__":
    main()
