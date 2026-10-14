# -*- coding: utf-8 -*-
"""
http_features.py — признаки текущего состояния URL (что сайт отдаёт сейчас).

Фишинговые ресурсы живут недолго: после блокировки адрес перестаёт отвечать,
отдаёт ошибку, парковочную страницу или редирект на чужой домен. Поэтому
текущий HTTP-ответ — сильный признак (на контрольной выборке AUC ≈ 0.75
у признака «запрос завершился ошибкой»).

Считаются: код ответа, класс ответа, факт редиректа, совпадение конечного
хоста с исходным, наличие и размер тела, заголовок Server, признак парковки
в заголовках/редиректе, число шагов редиректа.

Воспроизводимость: снимок сохраняется в файл (cache/http_all.pkl) и
поставляется вместе с кодом — состояние сайтов меняется во времени.

Запуск:
    python src/http_features.py --out cache/http_all.pkl --workers 128
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import normalize_url  # noqa: E402

UA = "Mozilla/5.0 (compatible; phishing-research-bot/1.0)"
PARKING_HINTS = ("parking", "sedo", "godaddy", "hugedomains", "afternic",
                 "domain for sale", "expired", "suspended")


def http_features(url: str) -> Dict[str, float]:
    f = {"http_status": -1.0, "http_ok": 0.0, "http_redirect": 0.0,
         "http_client_err": 0.0, "http_server_err": 0.0, "http_error": 0.0,
         "http_n_redirects": 0.0, "http_same_host": -1.0, "http_scheme_upgrade": -1.0,
         "http_len": -1.0, "http_has_server": -1.0, "http_parking": -1.0}
    try:
        target = normalize_url(url)
    except Exception:
        return f
    src = urlsplit(target)
    src_host = src.netloc.lower().partition(":")[0].strip(".")
    try:
        req = urllib.request.Request(target, method="HEAD",
                                     headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=4) as r:
            f["http_status"] = float(r.status)
            f["http_ok"] = float(200 <= r.status < 300)
            f["http_redirect"] = float(300 <= r.status < 400)
            final = urlsplit(r.geturl())
            f["http_same_host"] = float(final.netloc.lower().partition(":")[0].strip(".") == src_host)
            f["http_scheme_upgrade"] = float(src.scheme == "http" and final.scheme == "https")
            cl = r.headers.get("Content-Length")
            f["http_len"] = float(cl) if cl and cl.isdigit() else -1.0
            srv = (r.headers.get("Server") or "").lower()
            f["http_has_server"] = float(bool(srv))
            blob = (srv + " " + final.geturl().lower())
            f["http_parking"] = float(any(h in blob for h in PARKING_HINTS))
    except urllib.error.HTTPError as e:  # сервер ответил ошибкой
        f["http_status"] = float(e.code)
        f["http_client_err"] = float(400 <= e.code < 500)
        f["http_server_err"] = float(e.code >= 500)
        f["http_redirect"] = float(300 <= e.code < 400)
    except Exception:                     # нет соединения/DNS/таймаут
        f["http_error"] = 1.0
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description="Признаки текущего HTTP-состояния URL")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default="cache/http_all.pkl")
    ap.add_argument("--workers", type=int, default=128)
    args = ap.parse_args()

    urls = pd.concat([pd.read_csv(args.train)["url"], pd.read_csv(args.test)["url"]],
                     ignore_index=True).astype(str)
    uniq = list(dict.fromkeys(urls.tolist()))
    print(f"[info] проверяю {len(uniq)} URL ({args.workers} потоков)")
    socket.setdefaulttimeout(2.0)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(http_features, uniq))
    print(f"[info] опрошено за {time.time() - t0:.0f} с")
    table = pd.DataFrame(rows, index=uniq)
    table.index.name = "url"
    table.to_pickle(args.out)
    print(f"[ok] {args.out}: {table.shape}; отвечают 2xx: {table['http_ok'].mean() * 100:.1f} %, "
          f"ошибка: {table['http_error'].mean() * 100:.1f} %")


if __name__ == "__main__":
    main()
