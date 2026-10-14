# -*- coding: utf-8 -*-
"""
tls_features.py — признаки TLS-сертификата домена.

Дополняет DNS-признаки: у фишинговых доменов сертификаты, как правило, свежие
(выданы дни назад, часто бесплатные Let's Encrypt), тогда как у легитимных
доменов возраст сертификата больше, а цепочка выдана платными УЦ или
корпоративным PKI.

Считаются: доступность TLS, издатель, срок действия (дни с выдачи и до
истечения), число SAN-имён, признак самоподписанного сертификата, признак
совпадения CN с запрошенным хостом.

Воспроизводимость: как и для DNS, снимок сохраняется в файл и поставляется
с кодом (состояние сертификатов меняется во времени).

Запуск:
    python src/tls_features.py --out cache/tls_all.pkl --workers 128
"""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import ssl
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dns_features import host_of  # noqa: E402

PAID_CA = ("digicert", "globalsign", "sectigo", "comodo", "entrust", "godaddy",
           "thawte", "verisign", "geotrust", "rapidssl", "amazon", "microsoft",
           "apple", "google trust", "certum", "quovadis", "swisscom", "telekom")


def tls_features(host: str) -> Dict[str, float]:
    f = {
        "tls_ok": 0.0, "tls_days_old": -1.0, "tls_days_left": -1.0,
        "tls_valid_days": -1.0, "tls_n_san": -1.0, "tls_cn_match": -1.0,
        "tls_self_signed": -1.0, "tls_issuer_free": -1.0, "tls_issuer_paid": -1.0,
        "tls_is_letsencrypt": -1.0, "tls_wildcard": -1.0, "tls_tls13": -1.0,
        "tls_cert_fp": "", "tls_san_join": "",
    }
    if not host:
        return f
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((host, 443), timeout=4.0) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                f["tls_ok"] = 1.0
                f["tls_tls13"] = 1.0 if ssock.version() == "TLSv1.3" else 0.0
                der = ssock.getpeercert(binary_form=True)
    except Exception:
        return f
    # ВАЖНО: при verify_mode=CERT_NONE метод getpeercert() возвращает пустой
    # словарь, поэтому сертификат декодируется из DER встроенным декодером
    # CPython (ssl._ssl._test_decode_cert) через временный PEM-файл.
    cert = None
    if der:
        f["tls_cert_fp"] = hashlib.sha256(der).hexdigest()[:32]
        path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as fh:
                fh.write(ssl.DER_cert_to_PEM_cert(der))
                path = fh.name
            cert = ssl._ssl._test_decode_cert(path)  # type: ignore[attr-defined]
        except Exception:
            cert = None
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
    if not cert:
        return f

    def parse_dt(v):
        try:
            return datetime.strptime(v, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        except Exception:
            return None

    now = datetime.now(timezone.utc)
    nb, na = parse_dt(cert.get("notBefore", "")), parse_dt(cert.get("notAfter", ""))
    if nb:
        f["tls_days_old"] = float((now - nb).days)
    if na:
        f["tls_days_left"] = float((na - now).days)
    if nb and na:
        f["tls_valid_days"] = float((na - nb).days)

    san: List[str] = []
    for rtype, value in cert.get("subjectAltName", ()):  # type: ignore[misc]
        if rtype == "DNS":
            san.append(value)
    f["tls_n_san"] = float(len(san))
    f["tls_wildcard"] = float(any(s.startswith("*.") for s in san))

    subject = dict(x[0] for x in cert.get("subject", ()))  # type: ignore[arg-type]
    issuer = dict(x[0] for x in cert.get("issuer", ()))    # type: ignore[arg-type]
    cn = str(subject.get("commonName", "")).lower()
    issuer_all = " ".join(str(v).lower() for v in issuer.values())
    f["tls_cn_match"] = float(bool(cn) and (cn == host or host.endswith("." + cn)
                                            or (cn.startswith("*.") and host.endswith(cn[1:]))))
    f["tls_self_signed"] = float(bool(subject) and subject == issuer)
    f["tls_is_letsencrypt"] = float("let's encrypt" in issuer_all or "r3" == issuer.get("commonName", "").lower())
    f["tls_issuer_free"] = float(f["tls_is_letsencrypt"] or "zerossl" in issuer_all
                                 or "buypass" in issuer_all or "cpanel" in issuer_all)
    f["tls_issuer_paid"] = float(any(k in issuer_all for k in PAID_CA))
    # список SAN нужен для поиска доменов одной кампании (общий сертификат)
    try:
        sans = sorted({v.lower() for rt, v in cert.get("subjectAltName", ()) if rt == "DNS"})
        f["tls_san_join"] = "|".join(sans[:40])
    except Exception:
        f["tls_san_join"] = ""
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description="Признаки TLS-сертификатов")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default="cache/tls_all.pkl")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--only-alive", default="cache/dns_a.pkl",
                    help="pkl с dns_resolves, чтобы не проверять мёртвые домены")
    args = ap.parse_args()

    urls = pd.concat([pd.read_csv(args.train)["url"], pd.read_csv(args.test)["url"]],
                     ignore_index=True).astype(str)
    hosts = sorted({host_of(u) for u in urls})
    if args.only_alive and os.path.exists(args.only_alive):
        alive = pd.read_pickle(args.only_alive)
        hosts = [h for h in hosts if bool(alive["dns_resolves"].get(h, 0))]
    print(f"[info] проверяю TLS у {len(hosts)} живых доменов "
          f"({args.workers} потоков)")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(tls_features, hosts))
    print(f"[info] опрошено за {time.time() - t0:.0f} с")
    table = pd.DataFrame(rows, index=hosts)
    table.index.name = "host"
    table.to_pickle(args.out)
    ok = table["tls_ok"].mean() if len(table) else 0
    print(f"[ok] {args.out}: {table.shape}; TLS доступен у {ok * 100:.1f} % живых доменов")


if __name__ == "__main__":
    main()
