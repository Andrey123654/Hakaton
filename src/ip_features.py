# -*- coding: utf-8 -*-
"""
ip_features.py — признаки IP-адреса и автономной системы (ASN).

Дополняют сетевые признаки: важно не только «домен жив», но и где он
размещён. Фишинговые ресурсы часто живут на дешёвом или «пуленепробиваемом»
хостинге, на адресах, выделенных недавно, либо, наоборот, на массовых
облачных платформах (Cloudflare Pages, GitHub Pages и т.п.).

Источник данных — bulk-сервис Team Cymru (whois.cymru.com, порт 43),
который за один запрос отдаёт до тысячи адресов:
    AS | IP | BGP Prefix | CC | Registry | Allocated | AS Name

Считаются: номер AS, возраст выделения IP-префикса, страна и регистратура,
признаки облака/CDN/хостинга, «плохих» сетей, а также факт отсутствия
маршрута (адрес не анонсируется).

Воспроизводимость: снимок сохраняется в файл и поставляется с кодом.

Запуск:
    python src/ip_features.py --out cache/ip_asn.pkl --workers 192
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dns.resolver  # noqa: E402

from dns_features import host_of  # noqa: E402

CLOUD_ASN_HINTS = ("cloudflare", "amazon", "google", "microsoft", "fastly", "akamai",
                   "digitalocean", "linode", "ovh", "hetzner", "leaseweb", "scaleway",
                   "vultr", "contabo", "hostinger", "godaddy", "namecheap", "selectel",
                   "timeweb", "reg.ru", "beget", "yandex", "mail.ru", "oracle", "alibaba",
                   "tencent", "choopa", "colo", "choopa")
RESIDENTIAL_HINTS = ("comcast", "verizon", "at&t", "deutsche telekom", "orange",
                     "vodafone", "rostelecom", "beeline", "mts", "megafon", "sky",
                     "virgin", "bt ", "telefonica", "telecom italia")

RESOLVER = dns.resolver.Resolver(configure=False)
RESOLVER.nameservers = ["1.1.1.1", "8.8.8.8", "9.9.9.9"]
RESOLVER.lifetime = 1.5
RESOLVER.timeout = 1.5


def first_ip(host: str) -> str:
    try:
        ans = RESOLVER.resolve(host, "A", raise_on_no_answer=False)
        if ans.rrset:
            return str(list(ans.rrset)[0])
    except Exception:
        pass
    return ""


def cymru_bulk(ips: List[str], pause: float = 1.0) -> Dict[str, Dict[str, float]]:
    """Bulk-запрос к Team Cymru: до ~1000 адресов за одно соединение."""
    out: Dict[str, Dict[str, float]] = {}
    if not ips:
        return out
    payload = "begin\nverbose\n" + "\n".join(ips) + "\nend\n"
    try:
        s = socket.create_connection(("whois.cymru.com", 43), timeout=60)
        s.sendall(payload.encode())
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
        s.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Cymru недоступен: {exc}")
        return out
    now = datetime.now(timezone.utc)
    for line in data.decode("utf-8", "ignore").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 7 or not parts[0].isdigit():
            continue
        asn, ip, prefix, cc, registry, allocated, name = parts[:7]
        age = -1.0
        try:
            d = datetime.strptime(allocated, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            age = float((now - d).days)
        except Exception:
            age = -1.0
        low = name.lower()
        out[ip] = {
            "ip_asn": float(asn),
            "ip_age_days": age,
            "ip_asn_len": float(len(name)),
            "ip_cc_ru": float(cc.upper() in ("RU", "BY", "KZ", "UA")),
            "ip_cc_us": float(cc.upper() == "US"),
            "ip_cloud": float(any(h in low for h in CLOUD_ASN_HINTS)),
            "ip_residential": float(any(h in low for h in RESIDENTIAL_HINTS)),
            "ip_routed": 1.0,
        }
    time.sleep(pause)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Признаки IP/ASN по хостам")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default="cache/ip_asn.pkl")
    ap.add_argument("--workers", type=int, default=192)
    ap.add_argument("--chunk", type=int, default=900)
    args = ap.parse_args()

    urls = pd.concat([pd.read_csv(args.train)["url"], pd.read_csv(args.test)["url"]],
                     ignore_index=True).astype(str)
    hosts = sorted({host_of(u) for u in urls})
    print(f"[info] резолвлю {len(hosts)} хостов...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        ips_all = list(ex.map(first_ip, hosts))     # выровнено по hosts
    print(f"[info] получено IP: {sum(1 for x in ips_all if x)} за {time.time() - t0:.0f} с")

    uniq_ips = sorted({x for x in ips_all if x})
    print(f"[info] уникальных IP: {len(uniq_ips)}; запрашиваю ASN у Team Cymru...")
    info: Dict[str, Dict[str, float]] = {}
    for i in range(0, len(uniq_ips), args.chunk):
        part = uniq_ips[i:i + args.chunk]
        info.update(cymru_bulk(part))
        print(f"  {min(i + args.chunk, len(uniq_ips))}/{len(uniq_ips)}", flush=True)

    empty = {"ip_asn": -1.0, "ip_age_days": -1.0, "ip_asn_len": -1.0, "ip_cc_ru": -1.0,
             "ip_cc_us": -1.0, "ip_cloud": -1.0, "ip_residential": -1.0, "ip_routed": 0.0}
    rows = []
    for ip in ips_all:
        rows.append({**empty, **(info.get(ip, {}) if ip else {})})
    table = pd.DataFrame(rows, index=hosts)
    table.index.name = "host"
    table.to_pickle(args.out)
    print(f"[ok] {args.out}: {table.shape}; маршрутизируется "
          f"{table['ip_routed'].mean() * 100:.1f} % хостов")
    print(f"[info] всего времени: {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
