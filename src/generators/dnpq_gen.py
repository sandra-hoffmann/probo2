#!/usr/bin/env python3
"""
Generator fuer das Zufallsmodell D(n, p, q) aus

    Yong Gao. A Random Model for Argumentation Framework:
    Phase Transitions, Empirical Hardness, and Heuristics. IJCAI 2017, 503-509.

Definition 2.1: Die C(n,2) ungeordneten Argumentpaare entscheiden unabhaengig
voneinander ueber Existenz und Typ der Angriffsbeziehung.
  - mit Wahrscheinlichkeit p existiert eine Angriffsbeziehung
  - gegeben sie existiert, ist sie mit Wahrscheinlichkeit q symmetrisch,
    sonst einseitig mit gleichwahrscheinlicher Richtung

Pro Paar ergeben sich damit vier Ausgaenge:
    p*q            wechselseitig
    p*(1-q)/2      i -> j
    p*(1-q)/2      j -> i
    1-p            keine Kante

Theorem 1: fuer 0 < p, q < 1 gilt
    P(D(n,p,q) hat nicht-leere praeferierte Extension) -> 1  falls q > q*(p)
                                                       -> 0  falls q < q*(p)
mit q*(p) als eindeutiger Loesung von 4q / (1+q)^2 = p in (0,1).

Selbstangriffe kommen im Modell nicht vor (es wird ueber ungeordnete Paare
gezogen, nicht ueber n(n-1) gerichtete Kanten).

Beispiele
---------
  # Schwelle ausrechnen
  python3 dnpq_gen.py qstar --p 0.75

  # 100 Instanzen am beobachteten Uebergang (Gaos Figur 1/2 Setup)
  python3 dnpq_gen.py gen --n 1024 --p 0.75 --q 0.26 --count 100 --out ./bench

  # Scan um die theoretische Schwelle herum, 20 Instanzen je Punkt
  python3 dnpq_gen.py scan --n 512 --p 0.64 --span 0.12 --steps 13 \
      --count 20 --out ./scan

  # dichtegleiche Vega-Baseline (ER) zu p=0.75, q=0.26  ->  p_tilde = 0.4725
  python3 dnpq_gen.py gen --n 1024 --p 0.75 --q 0.26 --count 100 \
      --vega --out ./bench_er
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

# --------------------------------------------------------------------------
# Schwelle
# --------------------------------------------------------------------------


def p_of_qstar(q: float) -> float:
    """Schwellenkurve 4q/(1+q)^2. Streng monoton wachsend auf [0,1]."""
    return 4.0 * q / ((1.0 + q) ** 2)


def qstar(p: float) -> float:
    """Eindeutiges q* in (0,1) mit 4q*/(1+q*)^2 = p.

    Geschlossene Loesung von p*q^2 + (2p-4)*q + p = 0, kleinere Wurzel:
        q* = (2 - p - 2*sqrt(1-p)) / p
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p muss echt zwischen 0 und 1 liegen")
    return (2.0 - p - 2.0 * math.sqrt(1.0 - p)) / p


def p_tilde(p: float, q: float) -> float:
    """Kantenwahrscheinlichkeit des dichtegleichen Vega-Modells: p(1+q)/2.

    Das ist zugleich die Wahrscheinlichkeit, dass eine bestimmte gerichtete
    Kante in D(n,p,q) existiert.
    """
    return p * (1.0 + q) / 2.0


# --------------------------------------------------------------------------
# Generatoren
# --------------------------------------------------------------------------


def gen_dnpq(n: int, p: float, q: float, rng: random.Random) -> list[tuple[int, int]]:
    """Zieht eine Instanz aus D(n,p,q). Argumente sind 1..n.

    Ein einziger Uniform-Wurf pro Paar entscheidet Existenz, Typ und Richtung,
    indem [0,1) in vier Segmente zerlegt wird.
    """
    t_mutual = p * q
    t_forward = t_mutual + p * (1.0 - q) / 2.0
    t_backward = p  # Rest bis p ist die Gegenrichtung
    edges: list[tuple[int, int]] = []
    rand = rng.random
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            r = rand()
            if r >= t_backward:
                continue
            if r < t_mutual:
                edges.append((i, j))
                edges.append((j, i))
            elif r < t_forward:
                edges.append((i, j))
            else:
                edges.append((j, i))
    return edges


def gen_vega(n: int, pt: float, rng: random.Random) -> list[tuple[int, int]]:
    """Zieht eine Instanz aus Vegas Modell D~(n, p~).

    Jede der n(n-1) gerichteten Kanten unabhaengig mit Wahrscheinlichkeit p~.
    """
    edges: list[tuple[int, int]] = []
    rand = rng.random
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j and rand() < pt:
                edges.append((i, j))
    return edges


# --------------------------------------------------------------------------
# Ausgabeformate
# --------------------------------------------------------------------------


def to_i23(n: int, edges: list[tuple[int, int]], header: str) -> str:
    """ICCMA-2023-Format: 'p af <n>', danach eine Zeile je Angriff."""
    out = [f"p af {n}"]
    out.extend(f"{a} {b}" for a, b in edges)
    return "\n".join(out) + "\n"


def to_apx(n: int, edges: list[tuple[int, int]], header: str) -> str:
    """ASPARTIX-Format: arg(i). und att(i,j)."""
    out = [f"% {header}"]
    out.extend(f"arg({i})." for i in range(1, n + 1))
    out.extend(f"att({a},{b})." for a, b in edges)
    return "\n".join(out) + "\n"


WRITERS = {"i23": (to_i23, ".af"), "apx": (to_apx, ".apx")}


def write_batch(
    outdir: str,
    n: int,
    p: float,
    q: float,
    count: int,
    seed: int,
    fmt: str,
    vega: bool,
    quiet: bool = False,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    writer, ext = WRITERS[fmt]
    qs = qstar(p)
    pt = p_tilde(p, q)
    for k in range(count):
        rng = random.Random((seed, n, round(p, 6), round(q, 6), k, vega).__hash__())
        if vega:
            edges = gen_vega(n, pt, rng)
            stem = f"vega_n{n}_pt{pt:.4f}_seed{seed}_s{k:03d}"
            head = (
                f"Vega D~(n={n}, p~={pt:.6f}) | dichtegleich zu "
                f"D(n,p={p},q={q}) | seed={seed} rep={k}"
            )
        else:
            edges = gen_dnpq(n, p, q, rng)
            stem = f"dnpq_n{n}_p{p:.3f}_q{q:.4f}_seed{seed}_s{k:03d}"
            head = (
                f"Gao D(n={n}, p={p}, q={q}) | q*={qs:.6f} "
                f"({'above' if q > qs else 'below'} threshold) | "
                f"p~_equal_density={pt:.6f} | seed={seed} rep={k}"
            )
        path = os.path.join(outdir, stem + ext)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(writer(n, edges, head))
    if not quiet:
        tag = f"p~={pt:.4f}" if vega else f"q={q:.4f} (q*={qs:.4f})"
        print(f"  {count:4d} x n={n} {tag}  ->  {outdir}", file=sys.stderr)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def cmd_qstar(a: argparse.Namespace) -> None:
    if a.p is not None:
        qs = qstar(a.p)
        print(f"p = {a.p:.6f}   ->   q* = {qs:.6f}")
        print(f"  Probe: 4q*/(1+q*)^2 = {p_of_qstar(qs):.6f}")
        print(f"  dichtegleiches Vega-p~ bei q=q*: {p_tilde(a.p, qs):.6f}")
    else:
        print(f"q = {a.q:.6f}   ->   p* = {p_of_qstar(a.q):.6f}")


def cmd_gen(a: argparse.Namespace) -> None:
    q = a.q if a.q is not None else qstar(a.p) + a.offset
    if not 0.0 < q < 1.0:
        sys.exit(f"q = {q:.4f} liegt ausserhalb (0,1)")
    write_batch(a.out, a.n, a.p, q, a.count, a.seed, a.format, a.vega)


def cmd_scan(a: argparse.Namespace) -> None:
    qs = qstar(a.p)
    lo, hi = qs - a.span / 2.0, qs + a.span / 2.0
    step = (hi - lo) / (a.steps - 1) if a.steps > 1 else 0.0
    print(
        f"Scan um q*({a.p}) = {qs:.6f}, {a.steps} Punkte in "
        f"[{lo:.4f}, {hi:.4f}]",
        file=sys.stderr,
    )
    for s in range(a.steps):
        q = lo + s * step
        if not 0.0 < q < 1.0:
            print(f"  uebersprungen: q={q:.4f}", file=sys.stderr)
            continue
        sub = os.path.join(a.out, f"q{q:.4f}")
        write_batch(sub, a.n, a.p, q, a.count, a.seed, a.format, False)
        if a.with_vega:
            write_batch(
                os.path.join(a.out, f"q{q:.4f}_vega"),
                a.n, a.p, q, a.count, a.seed, a.format, True,
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generator fuer Gaos D(n,p,q)-Modell (IJCAI 2017)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("qstar", help="Schwelle q*(p) bzw. p*(q) ausrechnen")
    g = t.add_mutually_exclusive_group(required=True)
    g.add_argument("--p", type=float)
    g.add_argument("--q", type=float)
    t.set_defaults(func=cmd_qstar)

    common = dict(type=int, default=None)
    t = sub.add_parser("gen", help="Instanzen zu festem (n,p,q) erzeugen")
    t.add_argument("--n", required=True, **common)
    t.add_argument("--p", type=float, required=True)
    t.add_argument("--q", type=float, help="absolutes q; sonst --offset benutzen")
    t.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="q relativ zu q*(p), z.B. -0.05 fuer knapp unterhalb",
    )
    t.add_argument("--count", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--format", choices=list(WRITERS), default="i23")
    t.add_argument(
        "--vega",
        action="store_true",
        help="stattdessen dichtegleiche ER-Baseline D~(n, p(1+q)/2)",
    )
    t.add_argument("--out", default="./out")
    t.set_defaults(func=cmd_gen)

    t = sub.add_parser("scan", help="q-Scan um die Schwelle herum")
    t.add_argument("--n", type=int, required=True)
    t.add_argument("--p", type=float, required=True)
    t.add_argument("--span", type=float, default=0.1, help="Breite des q-Fensters")
    t.add_argument("--steps", type=int, default=11)
    t.add_argument("--count", type=int, default=20)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--format", choices=list(WRITERS), default="i23")
    t.add_argument(
        "--with-vega",
        action="store_true",
        help="je Punkt zusaetzlich dichtegleiche ER-Baseline erzeugen",
    )
    t.add_argument("--out", default="./scan")
    t.set_defaults(func=cmd_scan)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
