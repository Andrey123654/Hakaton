# -*- coding: utf-8 -*-
"""
dns_features.py — признаки DNS-состояния домена.

Идея и обоснование. Набор организаторов собран в 2024 году; фишинговые домены
живут недолго — их снимают по жалобам и блокируют, тогда как легитимные домены
продолжают работать. Поэтому «домен больше не резолвится» — сильный признак
фишинга (на контрольной выборке AUC ≈ 0.79 у одного этого признака).

Кроме факта резолвинга считаются:
  * число A/AAAA-адресов и признак попадания в облако/хостинг;
  * наличие MX (у фишинговых доменов почта обычно не настроена);
  * наличие NS и их «семейство» (Cloudflare, регистратор-парковщик и т.п.);
  * наличие TXT/SPF/DMARC (у легитимных доменов почти всегда есть);
  * наличие CNAME и признак парковки домена.

ВАЖНО про воспроизводимость: результат DNS-запросов меняется во времени.
Поэтому снимок признаков сохраняется в файл (cache/dns_<split>.pkl) и
поставляется вместе с кодом — воспроизведение опирается на этот снимок,
а не на повторные запросы в сеть.

Запуск:
    python src/dns_features.py --out cache/dns_all.pkl --workers 96
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import normalize_url, split_host  # noqa: E402

try:
    import dns.resolver
    HAVE_DNS = True
except Exception:  # pragma: no cover
    HAVE_DNS = False

# Прямой опрос публичных резолверов: системный резолвер macOS (mDNSResponder)
# сериализует параллельные запросы и на 55 тыс. доменов работает часами.
RESOLVER = None
if HAVE_DNS:
    RESOLVER = dns.resolver.Resolver(configure=False)
    RESOLVER.nameservers = ["1.1.1.1", "8.8.8.8", "9.9.9.9"]
    RESOLVER.lifetime = 1.5
    RESOLVER.timeout = 1.5

PARKING_NS = ("sedoparking", "parkingcrew", "above.com", "bodis", "dan.com",
              "afternic", "hugedomains", "namecheap", "parklogic", "trellian")
CLOUD_NS = ("cloudflare", "awsdns", "azure-dns", "googledomains", "domaincontrol",
            "nsone", "dnsimple", "digitalocean", "linode", "yandex", "reg.ru",
            "timeweb", "beget", "sprinthost", "masterhost", "nic.ru")


def host_of(url: str) -> str:
    h = urlsplit(normalize_url(url)).netloc.lower()
    if "@" in h:
        h = h.rsplit("@", 1)[1]
    return h.partition(":")[0].strip(".")


def query(name: str, rdtype: str) -> List[str]:
    """DNS-запрос через публичный резолвер; пустой список при любой ошибке."""
    if RESOLVER is None:
        return []
    try:
        ans = RESOLVER.resolve(name, rdtype, raise_on_no_answer=False)
        if ans.rrset is None:
            return []
        return [str(r).strip('"') for r in ans]
    except Exception:
        return []


def resolve_host(host: str) -> Dict[str, float]:
    """Фаза 1: A/AAAA-записи (самый сильный признак) — быстрый проход."""
    f = {"dns_host_len": float(len(host))}
    if not host:
        return {**f, "dns_resolves": 0.0, "dns_n_ips": 0.0, "dns_n_ipv6": 0.0}
    a = query(host, "A")
    aaaa = query(host, "AAAA")
    ips = a + aaaa
    ipv4 = list(a)
    f["dns_resolves"] = float(bool(ips))
    f["dns_n_ips"] = float(len(ipv4))
    f["dns_n_ipv6"] = float(len(ips) - len(ipv4))
    f["_first_ip"] = ipv4[0] if ipv4 else ""
    return f


def dns_features(host: str, resolved: bool = True, first_ip: str = "") -> Dict[str, float]:
    """
    Фаза 2: MX/NS/TXT/DMARC/CNAME/PTR. Для неразрешающихся доменов запросы
    не выполняются — это кратно ускоряет обработку (фишинговые домены мертвы).
    """
    f: Dict[str, float] = {}
    if not resolved:
        return {"dns_has_mx": 0.0, "dns_n_mx": 0.0, "dns_mx_google": 0.0,
                "dns_mx_microsoft": 0.0, "dns_mx_yandex": 0.0, "dns_has_ns": 0.0,
                "dns_n_ns": 0.0, "dns_ns_cloud": 0.0, "dns_ns_parking": 0.0,
                "dns_has_txt": 0.0, "dns_has_spf": 0.0, "dns_has_dmarc": 0.0,
                "dns_has_cname": 0.0, "dns_has_ptr": 0.0}
    mx = query(host, "MX")
    f["dns_has_mx"] = float(bool(mx))
    f["dns_n_mx"] = float(len(mx))
    mx_join = " ".join(mx).lower()
    f["dns_mx_google"] = float("google" in mx_join or "gmail" in mx_join)
    f["dns_mx_microsoft"] = float("outlook" in mx_join or "microsoft" in mx_join)
    f["dns_mx_yandex"] = float("yandex" in mx_join or "mail.ru" in mx_join)
    ns = query(host, "NS")
    f["dns_has_ns"] = float(bool(ns))
    f["dns_n_ns"] = float(len(ns))
    ns_join = " ".join(ns).lower()
    f["dns_ns_cloud"] = float(any(k in ns_join for k in CLOUD_NS))
    f["dns_ns_parking"] = float(any(k in ns_join for k in PARKING_NS))
    txt = query(host, "TXT")
    f["dns_has_txt"] = float(bool(txt))
    f["dns_has_spf"] = float(any("v=spf1" in t.lower() for t in txt))
    dmarc = query(f"_dmarc.{host}", "TXT")
    f["dns_has_dmarc"] = float(any("v=dmarc1" in t.lower() for t in dmarc))
    f["dns_has_cname"] = float(bool(query(host, "CNAME")))
    ptr = 0.0
    if first_ip:
        try:
            socket.gethostbyaddr(first_ip)
            ptr = 1.0
        except Exception:
            ptr = 0.0
    f["dns_has_ptr"] = ptr
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description="Признаки DNS-состояния доменов")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default="cache/dns_all.pkl")
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()

    urls = pd.concat([pd.read_csv(args.train)["url"], pd.read_csv(args.test)["url"]],
                     ignore_index=True).astype(str)
    hosts = sorted({host_of(u) for u in urls})
    print(f"[info] уникальных хостов: {len(hosts)} "
          f"(dnspython: {'есть' if HAVE_DNS else 'нет'})")

    t0 = time.time()
    print("[info] фаза 1/2: A/AAAA-записи (прямой резолвер)...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers * 2) as ex:
        phase1 = list(ex.map(resolve_host, hosts))
    p1 = pd.DataFrame(phase1, index=hosts)
    alive = p1["dns_resolves"] > 0
    print(f"[info]   живых доменов: {int(alive.sum())} из {len(hosts)} "
          f"({time.time() - t0:.0f} с)", flush=True)
    print("[info] фаза 2/2: MX/NS/TXT/DMARC/CNAME для живых доменов...", flush=True)
    alive_hosts = [h for h in hosts if alive.loc[h]]
    first_ips = {h: str(p1.loc[h, "_first_ip"]) for h in alive_hosts}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        phase2 = list(ex.map(lambda h: dns_features(h, True, first_ips.get(h, "")), alive_hosts))
    p2 = pd.DataFrame(phase2, index=alive_hosts) if phase2 else pd.DataFrame(index=alive_hosts)
    dead = p1.loc[~alive].drop(columns=["_first_ip"])
    dead = pd.concat([dead, pd.DataFrame([dns_features("", False)] * len(dead), index=dead.index)], axis=1)
    alive_tbl = pd.concat([p1.loc[alive].drop(columns=["_first_ip"]), p2], axis=1)
    table = pd.concat([alive_tbl, dead]).loc[hosts]
    print(f"[info] опрошено за {time.time() - t0:.0f} с")
    table.index.name = "host"
    table.to_pickle(args.out)
    print(f"[ok] {args.out}: {table.shape}; резолвится {table['dns_resolves'].mean() * 100:.1f} % хостов")


if __name__ == "__main__":
    main()
