#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbital.py -- The Orbital game, version 0.1 (Z-Botvinnik, 2026-10-15)

What this version does
    1. Reads a day file  day_YYYY-MM-DD-N.txt  (grammar: see parse_day()).
    2. MC1: vessel orbit + home-ship orbit + kappa + L, from DAY and SILENCE.
    3. MC2: junky-station orbit from its two encounter times.
       Both posteriors are those of the notes (with the corrections listed below).
       Modes are found first (docking/encounters solved exactly as 2D Lambert
       problems, then a Nelder-Mead MAP step); emcee then samples the exact
       posterior around each mode. Blind emcee from the prior cannot find them:
       the constraints define isolated points in parameter space.
    4. A matplotlib window with a slider over the posterior samples (sorted by
       log posterior, lowest -> highest) and a Save button.  Save writes the
       SAVED PARAMETERS block back into the day file (a .bak copy is kept).
    5. Day view: orbits coloured by local time (cyclic HSV, 0 h = blue), real
       silences, junky station, container end times (MC3 placeholder),
       prophets, reminders, earnings, and the current positions.
    6. Watch loop: re-reads the file every --interval seconds.  A stage is
       rerun only when its SAVED PARAMETERS lines are missing.
    7. Reminders: times drawn from NORMAL [mu, sd] minutes; spoken with the
       Windows voice named in VOICE "..." (console print elsewhere).

Not in this version: MC3 (container launch orbits), planet rotation W,
HTML/ECharts, earthquake probes, beauty functions.

Corrections to the notes (see occluded(), orbit_ok(), CONFIG['f_is_kappa_marginal'])
    a. Occlusion uses the SEGMENT between the bodies: x^2 < 1 AND 0 < s* < 1.
       x is the distance to the infinite line; with x alone, two bodies on the same
       ray (e.g. at launch) count as silent.
    b. No collision needs q = p/(1 + e) >= 1; lambda^2 >= kappa and eta >= -kappa/2
       are necessary, not sufficient.  e^2 >= 0 must also hold.
    c. Ellipse inside L needs Q = p/(1 - e) <= L; (lambda^2/(-2 eta))^(1/2) is the
       semi-minor axis b/R, not the apocentre.
    d. Flat (eta, lambda) priors have volume ~ kappa^(3/2) per orbit, so pi(Y) as
       written gives kappa an effective prior f * kappa^3; divided out by default.

Dimensionless units (as in the notes)
    tau = (t - t_start)/T,  u = r/R,
    eta = (E/mu) T^2/R^2,  lambda = (l/mu) T/R^2,  kappa = (k/mu) T^2/R^3
    kappa = (4/3) pi G rho T^2   (m << M)

Usage
    python orbital.py day_2026-10-15-1.txt                # watch mode, 30 s
    python orbital.py day_2026-10-15-1.txt --once         # one pass, then show day view
    python orbital.py day_2026-10-15-1.txt --auto         # take best model, no slider
    python orbital.py day_2026-10-15-1.txt --no-voice
    python orbital.py day_2026-10-15-1.txt --snapshot day.png   # save day view and exit

Requires: numpy, matplotlib, emcee (pip install emcee)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# =============================================================================
# CONFIGURATION (edit here)
# =============================================================================
CONFIG = dict(
    # --- MC1 likelihood widths -------------------------------------------------
    sigma_tau=0.005,      # silence mismatch, fraction of T  (0.005 * 12 h = 3.6 min)
    sigma_xi0=0.02,       # radial mismatch at launch (tau = 0), units of R
    sigma_xi1=0.02,       # vessel-home ship distance at return (tau = 1), units of R
    # --- MC2 likelihood widths -------------------------------------------------
    sigma_js=0.02,        # vessel-station distance at both encounters, units of R
    # --- priors ------------------------------------------------------------------
    rho_min=250.0,        # kg/m^3   f(kappa/T^2): log-uniform in rho on [rho_min, rho_max]
    rho_max=25000.0,      # kg/m^3   (hot Jupiters ... red dwarfs; no white dwarfs)
    c1=1.0 / 25.0,         # L prior  g(L) ~ exp(-c1 L)
    allow_retrograde=False,  # lambda < 0 allowed (orbit sense free); False -> prograde only
    f_is_kappa_marginal=True,  # True: divide pi(Y) by kappa^3 so that f is the marginal prior
                               # of kappa (flat eta, lambda boxes have volume ~ kappa^(3/2) each).
                               # False: pi(Y) exactly as written in the notes.
    # --- mode search (exact docking via Lambert) + emcee around each mode -------------
    n_grid=1000,          # tau grid for the silence integral
    n_grid_screen=400,    # coarser grid used while screening
    search_rounds=((2000, None), (1500, 0.03), (1500, 0.006)),  # (vessel draws, local scale)
    search_top=40,        # vessels kept between rounds
    n_refine=30,          # distinct candidates given a MAP (Nelder-Mead) refinement
    n_modes=12,           # distinct modes handed to emcee
    walkers_per_mode=20,
    steps_per_mode=600,   # second half kept
    n_modes_mc2=16,
    walkers_per_mode_mc2=8,
    steps_per_mode_mc2=600,
    n_show=300,           # samples offered on the slider
    seed=None,            # int for reproducible runs
    # --- display -------------------------------------------------------------------
    R_km=None,            # planet radius in km (lore); None -> lengths shown in R
)

G_SI = 6.67430e-11
TWO_PI = 2.0 * np.pi
KNOWN_PRIZES = ("EMERALD", "SAPPHIRE", "DIAMOND", "BALL", "MARTIAN")
EMOJI = {"EMERALD": "\U0001F7E2", "SAPPHIRE": "\U0001F535", "DIAMOND": "⚪",
         "BALL": "⚽", "MARTIAN": "\U0001FA99"}
PLURAL = {"EMERALD": "emeralds", "SAPPHIRE": "sapphires", "DIAMOND": "diamonds",
          "BALL": "Lian balls", "MARTIAN": "Martian coins"}


# =============================================================================
# CLOCK UTILITIES
# =============================================================================
CLOCK = r"\d{1,2}:\d{2}(?::\d{2})?"
INTERVAL_RE = re.compile(r"\[\s*(%s)\s*(?:-\s*(%s)\s*)?\]" % (CLOCK, CLOCK))


def clock_to_s(s: str) -> int:
    parts = [int(x) for x in s.split(":")]
    h, m = parts[0], parts[1]
    sec = parts[2] if len(parts) > 2 else 0
    if not (0 <= h <= 24 and 0 <= m < 60 and 0 <= sec < 60):
        raise ValueError(f"bad clock time '{s}'")
    return h * 3600 + m * 60 + sec


def s_to_clock(s: float) -> str:
    s = int(round(s)) % 86400
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def find_intervals(text: str):
    """All [a - b] or [a] groups in text -> list of (a_s, b_s or None)."""
    out = []
    for m in INTERVAL_RE.finditer(text):
        a = clock_to_s(m.group(1))
        b = clock_to_s(m.group(2)) if m.group(2) else None
        out.append((a, b))
    return out


# =============================================================================
# DAY-FILE MODEL AND PARSER
# =============================================================================
@dataclass
class Emerald:
    text: str
    done: bool
    done_at: str | None
    prizes: Counter


@dataclass
class Reminder:
    name: str
    start: int
    end: int
    mu: float
    sd: float
    voice: str
    messages: list = field(default_factory=list)
    block: str | None = None


@dataclass
class Block:
    kind: str          # "JUNKY STATION" | "CONTAINER"
    name: str
    start: int
    end: int
    emeralds: list = field(default_factory=list)
    why: list = field(default_factory=list)
    report: str = ""


@dataclass
class DayFile:
    path: Path | None
    date: dt.date
    saved: dict = field(default_factory=dict)
    day: tuple | None = None          # (start_s, end_s) clock seconds
    silences: list = field(default_factory=list)
    prophets: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    reminders: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def T(self) -> float:
        return day_length(self.day)


def day_length(day) -> float:
    a, b = day
    T = (b - a) % 86400
    return float(T)


def clock_offset(clock_s: float, day) -> float:
    """Seconds from day start to a clock time.  Clock times in the first half of
    the off-shift gap count as after the day, in the second half as before it."""
    T = day_length(day)
    off = (clock_s - day[0]) % 86400
    if off > T + 0.5 * (86400 - T):
        off -= 86400
    return float(off)


def clock_tau(clock_s: float, day) -> float:
    return clock_offset(clock_s, day) / day_length(day)


def tau_clock(tau: float, day) -> str:
    return s_to_clock(day[0] + tau * day_length(day))


def date_from_name(path: Path | None) -> dt.date:
    if path is not None:
        m = re.search(r"day_(\d{4})-(\d{2})-(\d{2})", path.name)
        if m:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return dt.date.today()


EMERALD_RE = re.compile(r"^\*E(d?)(\d{1,2}:\d{2})?(?=\s|$)\s*(.*)$")
BULLET_RE = re.compile(r"^\*\s*(.*)$")
JS_RE = re.compile(r"^JUNKY\s+STATION\s+(\S+)\s*(\[.*)$")
CONT_RE = re.compile(r"^CONTAINER\s+(\S+)\s*(\[.*)$")
REM_RE = re.compile(r"^REMINDER\s+(\S+)\s*(\[[^\]]*\])(.*)$")
NORMAL_RE = re.compile(r"NORMAL\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")
VOICE_RE = re.compile(r'VOICE\s*"([^"]*)"')
PRIZE_TOKEN_RE = re.compile(r"^(\d+)?\s*([A-Za-z_]+)$")


def parse_prizes(s: str, warn, ln) -> Counter:
    c = Counter()
    for tok in s.split("+"):
        tok = tok.strip()
        if not tok:
            continue
        m = PRIZE_TOKEN_RE.match(tok)
        if not m:
            warn(f"line {ln}: prize '{tok}' not understood")
            continue
        n = int(m.group(1)) if m.group(1) else 1
        noun = m.group(2).upper()
        if noun not in KNOWN_PRIZES and noun.endswith("S") and noun[:-1] in KNOWN_PRIZES:
            noun = noun[:-1]
        if noun not in KNOWN_PRIZES:
            warn(f"line {ln}: unknown prize '{noun}' (kept)")
        c[noun] += n
    return c


def parse_saved_line(line: str):
    line = line.split("#", 1)[0].strip()
    if "=" not in line:
        return None, None
    k, v = line.split("=", 1)
    return k.strip(), v.strip()


def parse_day(text: str, path: Path | None = None) -> DayFile:
    """Grammar (one keyword per line, keywords in capitals):
        SAVED PARAMETERS ... END SAVED PARAMETERS     KEY = value lines (program-written)
        DAY [hh:mm - hh:mm]
        SILENCE [a - b], [c - d], ...
        PROPHETS [t1], [t2], ...
        JUNKY STATION <name> [a - b]                  starts a block
        CONTAINER <name> [a - b]                      starts a block
          *E text            emerald                  *Ed text      emerald done
          *EdHH:MM text      done at HH:MM            ... ===> SAPPHIRE + 3 MARTIANS
          WHY                then '* ...' bullets
          REPORT             then a \"\"\" ... \"\"\" block
        REMINDER <name> [a - b], NORMAL [mu, sd], VOICE "Zira"
          * "message"        one or more
        END REMINDER
        END DAY                                       everything after is ignored
    """
    d = DayFile(path=path, date=date_from_name(path))
    warn = d.warnings.append
    mode = None            # None | saved | why | report_wait | report_body | reminder
    block = None
    rem = None
    report_lines = []

    def close_report():
        nonlocal report_lines
        if block is not None and report_lines:
            block.report = "\n".join(report_lines).strip()
        report_lines = []

    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()

        if mode == "saved":
            if line.startswith("END SAVED PARAMETERS"):
                mode = None
            else:
                k, v = parse_saved_line(line)
                if k:
                    d.saved[k] = v
            continue

        if mode == "report_body":
            if '"""' in line:
                head = raw.split('"""', 1)[0].rstrip()
                if head.strip():
                    report_lines.append(head)
                close_report()
                mode = None
            else:
                report_lines.append(raw.rstrip())
            continue

        if not line or line.strip('"') == "":
            continue

        if mode == "report_wait" and line.startswith('"""'):
            rest = line[3:]
            if '"""' in rest:
                report_lines.append(rest.split('"""', 1)[0])
                close_report()
                mode = None
            else:
                if rest.strip():
                    report_lines.append(rest)
                mode = "report_body"
            continue

        # ---------------------------------------------------------------- keywords
        if line.startswith("SAVED PARAMETERS"):
            if mode == "report_wait":
                close_report()
            mode = "saved"
            continue
        if line.startswith("END DAY"):
            if mode == "report_wait":
                close_report()
            break
        if line.startswith("END REMINDER"):
            rem = None
            mode = None
            continue

        is_kw = re.match(r"^(DAY|SILENCE|PROPHETS|JUNKY\s+STATION|CONTAINER|REMINDER|WHY|REPORT)\b", line)
        if is_kw and mode == "report_wait":
            close_report()
            mode = None

        if re.match(r"^DAY\b", line):
            iv = find_intervals(line)
            if len(iv) != 1 or iv[0][1] is None:
                warn(f"line {ln}: DAY needs one interval [a - b]")
            else:
                d.day = iv[0]
                if day_length(d.day) == 0:
                    warn(f"line {ln}: DAY has zero length")
                    d.day = None
            continue
        if re.match(r"^SILENCE\b", line):
            for a, b in find_intervals(line):
                if b is None:
                    warn(f"line {ln}: SILENCE entry [{s_to_clock(a)}] is not an interval")
                else:
                    d.silences.append((a, b))
            continue
        if re.match(r"^PROPHETS\b", line):
            d.prophets.extend(a for a, _ in find_intervals(line))
            continue
        m = JS_RE.match(line) or CONT_RE.match(line)
        if m:
            kind = "JUNKY STATION" if line.startswith("JUNKY") else "CONTAINER"
            iv = find_intervals(m.group(2))
            if len(iv) != 1 or iv[0][1] is None:
                warn(f"line {ln}: {kind} {m.group(1)} needs one interval [a - b]")
                block = Block(kind, m.group(1), 0, 0)
                block.start = block.end = -1
            else:
                block = Block(kind, m.group(1), iv[0][0], iv[0][1])
            d.blocks.append(block)
            mode, rem = None, None
            continue
        m = REM_RE.match(line)
        if m:
            iv = find_intervals(m.group(2))
            if len(iv) != 1 or iv[0][1] is None:
                warn(f"line {ln}: REMINDER {m.group(1)} needs one interval [a - b]")
                mode = "reminder"
                rem = None
                continue
            nm = NORMAL_RE.search(m.group(3))
            vm = VOICE_RE.search(m.group(3))
            mu, sd = (float(nm.group(1)), float(nm.group(2))) if nm else (10.0, 1.0)
            if not nm:
                warn(f"line {ln}: REMINDER {m.group(1)} without NORMAL [mu, sd]; using [10, 1]")
            rem = Reminder(m.group(1), iv[0][0], iv[0][1], mu, sd,
                           vm.group(1) if vm else "Zira", [],
                           block.name if block is not None else None)
            d.reminders.append(rem)
            mode = "reminder"
            continue
        if line == "WHY" or line.startswith("WHY "):
            mode = "why"
            continue
        if line.startswith("REPORT"):
            rest = line[len("REPORT"):].strip()
            report_lines = []
            if rest.startswith('"""'):
                rest = rest[3:]
                if '"""' in rest:
                    report_lines.append(rest.split('"""', 1)[0])
                    close_report()
                    mode = None
                else:
                    if rest.strip():
                        report_lines.append(rest)
                    mode = "report_body"
            else:
                if rest:
                    report_lines.append(rest)
                mode = "report_wait"
            continue

        # ---------------------------------------------------------------- items
        m = EMERALD_RE.match(line)
        if m:
            if mode == "reminder":
                warn(f"line {ln}: emerald inside REMINDER (missing END REMINDER?)")
            body = m.group(3)
            prizes = Counter()
            if "===>" in body:
                body, pz = body.split("===>", 1)
                prizes = parse_prizes(pz, warn, ln)
            em = Emerald(body.strip(), m.group(1) == "d", m.group(2), prizes)
            if block is None:
                warn(f"line {ln}: emerald outside any CONTAINER / JUNKY STATION (ignored)")
            else:
                block.emeralds.append(em)
            if mode == "why":
                mode = None
            continue
        m = BULLET_RE.match(line)
        if m:
            txt = m.group(1).strip()
            if mode == "reminder" and rem is not None:
                rem.messages.append(txt.strip('"').strip())
            elif mode == "why" and block is not None:
                block.why.append(txt)
            elif mode == "report_wait":
                report_lines.append(line)
            continue
        if mode == "report_wait":
            report_lines.append(line)
            continue
        if re.match(r"^[A-Z]{3,}(\s+[A-Z]{3,})*\s*\[", line):
            warn(f"line {ln}: unknown keyword line ignored: {line[:50]}")

    if mode == "report_wait":
        close_report()
    if mode == "report_body":
        warn("REPORT block without closing \"\"\"")
        close_report()
    if mode == "saved":
        warn("SAVED PARAMETERS without END SAVED PARAMETERS")
    for r in d.reminders:
        if not r.messages:
            warn(f"REMINDER {r.name} has no '* \"message\"' lines")
    validate(d)
    return d


def validate(d: DayFile):
    warn = d.warnings.append
    if d.day is None:
        warn("no valid DAY line: nothing can be computed")
        return
    for a, b in d.silences:
        ta, tb = clock_tau(a, d.day), clock_tau(b, d.day)
        if ta < 0 or tb > 1 or tb <= ta:
            warn(f"SILENCE [{s_to_clock(a)} - {s_to_clock(b)}] is not inside DAY; clipped")
        if ta <= 0.0 or tb >= 1.0:
            warn(f"SILENCE [{s_to_clock(a)} - {s_to_clock(b)}] touches the day start/end. "
                 "The vessel is docked at tau = 0 and tau = 1, so it cannot be occluded there; "
                 "the fit will end/start this silence earlier/later.")
    for b in d.blocks:
        if b.start < 0:
            continue
        ta, tb = clock_tau(b.start, d.day), clock_tau(b.end, d.day)
        if ta < 0 or tb > 1:
            warn(f"{b.kind} {b.name} [{s_to_clock(b.start)} - {s_to_clock(b.end)}] is outside "
                 f"DAY [{s_to_clock(d.day[0])} - {s_to_clock(d.day[1])}]")
    for p in d.prophets:
        if not 0 <= clock_tau(p, d.day) <= 1:
            warn(f"PROPHETS [{s_to_clock(p)}] is outside DAY (not shown on the orbit)")
    js = [b for b in d.blocks if b.kind == "JUNKY STATION"]
    if len(js) > 1:
        warn(f"{len(js)} JUNKY STATION blocks; MC2 uses the first ({js[0].name})")


def silences_tau(d: DayFile):
    out = []
    for a, b in d.silences:
        ta = min(max(clock_tau(a, d.day), 0.0), 1.0)
        tb = min(max(clock_tau(b, d.day), 0.0), 1.0)
        if tb > ta:
            out.append((ta, tb))
    return out


# =============================================================================
# TWO-BODY PHYSICS (dimensionless; vectorised)
# =============================================================================
def wrap_pi(x):
    return (x + np.pi) % TWO_PI - np.pi


def kappa_from_rho(rho, T_s):
    return 4.0 / 3.0 * np.pi * G_SI * rho * T_s ** 2


def rho_from_kappa(kappa, T_s):
    return kappa / (4.0 / 3.0 * np.pi * G_SI * T_s ** 2)


def orbit_elements(eta, lam, kappa):
    """p = alpha/R, e, a/R, mean motion n (radians per unit tau)."""
    p = lam ** 2 / kappa
    e = np.sqrt(np.clip(1.0 + 2.0 * eta * lam ** 2 / kappa ** 2, 0.0, None))
    a = kappa / (-2.0 * eta)
    n = (-2.0 * eta) ** 1.5 / kappa
    return p, e, a, n


def orbit_ok(eta, lam, kappa, L):
    """Bound ellipse with pericentre >= 1 (no collision) and apocentre <= L."""
    with np.errstate(all="ignore"):
        e2 = 1.0 + 2.0 * eta * lam ** 2 / kappa ** 2
        good = (eta < 0) & (e2 >= 0) & (lam != 0)
        e = np.sqrt(np.clip(e2, 0.0, None))
        p = lam ** 2 / kappa
        good &= (p / (1.0 + e) >= 1.0 - 1e-9) & (p / (1.0 - e) <= L * (1.0 + 1e-9))
    return good


def geo_to_phys(s1, w, kappa, L):
    """Sampling coordinates -> (eta, lambda) and log|Jacobian|.
    q = 1 + (L - 1) s1           pericentre,  s1 in [0, 1]
    Q = q + (L - q) |w|          apocentre,   |w| in (0, 1],  sign(lambda) = sign(w)
    eta = -kappa/(q + Q),  lambda^2 = 2 kappa q Q/(q + Q)
    |d(eta, lambda)/d(q, Q)| = kappa |lambda| (Q - q) / (2 q Q (q + Q)^2)
    |d(q, Q)/d(s1, |w|)|     = (L - 1)(L - q)"""
    with np.errstate(all="ignore"):
        q = 1.0 + (L - 1.0) * s1
        Q = q + (L - q) * np.abs(w)
        S = q + Q
        eta = -kappa / S
        lam = np.sign(w) * np.sqrt(2.0 * kappa * q * Q / S)
        logj = (np.log(kappa) + np.log(np.abs(lam)) + np.log(Q - q) - np.log(2.0 * q * Q) - 2.0 * np.log(S)
                + np.log(L - 1.0) + np.log(L - q))
    return eta, lam, logj


def phys_to_geo(eta, lam, kappa, L):
    """Inverse of geo_to_phys: (eta, lambda) -> (s1, w)."""
    p = lam ** 2 / kappa
    e = np.sqrt(np.clip(1.0 + 2.0 * eta * lam ** 2 / kappa ** 2, 0.0, None))
    q, Q = p / (1.0 + e), p / (1.0 - e)
    s1 = (q - 1.0) / (L - 1.0)
    w = np.sign(lam) * np.clip((Q - q) / (L - q), 1e-9, 1.0)
    return np.clip(s1, 0.0, 1.0), w


def lambert_2d(u0, u1, dth, dtau, kappa, L, allow_retro=True, n_grid=400):
    """All bound orbits with pericentre >= 1 and apocentre <= L that are at polar
    position (u0, 0) at tau = 0 and at (u1, dth) at tau = dtau, for every number of
    complete revolutions.  Returns [(eta, lambda, theta0)].

    Conics with focus at the origin:  1/u = A + B cos(theta) + C sin(theta),
    A = 1/p, (B, C) = (e/p)(cos w, sin w).  Through both points:
        B = 1/u0 - A,   C = (1/u1 - A - B cos dth)/sin dth     (linear in A)
    With b0 = 1/u0, c0 = (1/u1 - b0 cos dth)/sin dth, c1 = (cos dth - 1)/sin dth:
        B = b0 - A,  C = c0 + c1 A,  and  e < 1  <=>  c1^2 A^2 + 2 (c0 c1 - b0) A + b0^2 + c0^2 < 0,
    an interval (A-, A+) where tof -> infinity at both ends (parabolic limits).
    For each direction and revolution count N the time of flight
        tof(A) = (Delta M + 2 pi N)/n,   n = (1 - e^2)^(3/2) sqrt(kappa/p^3)
    is scanned on a grid clustered at both ends; roots of tof = dtau are bracketed and refined."""
    from scipy.optimize import brentq, minimize_scalar
    sn, cs = np.sin(dth), np.cos(dth)
    if abs(sn) < 1e-9:
        return []
    b0 = 1.0 / u0
    c0 = (1.0 / u1 - b0 * cs) / sn
    c1 = (cs - 1.0) / sn
    qa, qb, qc = c1 ** 2, 2.0 * (c0 * c1 - b0), b0 ** 2 + c0 ** 2
    disc = qb * qb - 4.0 * qa * qc
    if qa < 1e-14 or disc <= 0.0:
        return []
    A_lo = max((-qb - np.sqrt(disc)) / (2.0 * qa), 1.0 / L)
    A_hi = min((-qb + np.sqrt(disc)) / (2.0 * qa), 1.0)
    if A_lo >= A_hi:
        return []

    def conic(A):
        B = 1.0 / u0 - A
        C = (1.0 / u1 - A - B * cs) / sn
        p = 1.0 / A
        return p, p * np.hypot(B, C), np.arctan2(C, B)

    def dM_n(A, sd):
        p, e, om = conic(A)
        with np.errstate(all="ignore"):
            ok = e < 1.0                      # tof continuous on e < 1; q, Q limits checked at roots
            ee = np.where(e < 1.0, e, 0.5)
            dM = (mean_from_true(sd * (dth - om), ee) - mean_from_true(sd * (0.0 - om), ee)) % TWO_PI
            n = (1.0 - ee ** 2) ** 1.5 * np.sqrt(kappa / p ** 3)
        return dM, n, ok

    t = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]
    A = A_lo + (A_hi - A_lo) * 0.5 * (1.0 - np.cos(np.pi * t))
    out = []
    for sd in ((1.0, -1.0) if allow_retro else (1.0,)):
        dM, n, ok = dM_n(A, sd)
        if not ok.any():
            continue
        Nmax = int(np.max(n[ok]) * dtau / TWO_PI) + 1
        pair = ok[:-1] & ok[1:] & (np.abs(np.diff(dM)) < np.pi)
        for N in range(Nmax + 1):
            g = (dM + TWO_PI * N) / n - dtau

            def gf(a, N=N, sd=sd):
                dm, nn, _ = dM_n(np.array([a]), sd)
                return (dm[0] + TWO_PI * N) / nn[0] - dtau

            brackets = [(A[i], A[i + 1]) for i in np.flatnonzero(pair & (np.sign(g[:-1]) != np.sign(g[1:])))]
            # near-tangent roots: grid local minima of g that are slightly positive
            loc = np.flatnonzero((g[1:-1] > 0) & (g[1:-1] < g[:-2]) & (g[1:-1] < g[2:])
                                 & pair[:-1] & pair[1:] & (g[1:-1] < 0.05 * dtau)) + 1
            roots = []
            for i in loc:
                res = minimize_scalar(gf, bounds=(A[i - 1], A[i + 1]), method="bounded",
                                      options=dict(xatol=1e-15))
                if res.fun < 0.0:
                    brackets += [(A[i - 1], res.x), (res.x, A[i + 1])]
                elif res.fun < 1e-10 * dtau:
                    roots.append(res.x)
            for lo, hi in brackets:
                try:
                    roots.append(brentq(gf, lo, hi, xtol=1e-15, rtol=1e-13))
                except ValueError:
                    continue
            for a in roots:
                p, e, om = conic(a)
                if not (e < 1.0 and p / (1.0 + e) >= 1.0 and p / (1.0 - e) <= L):
                    continue
                eta = -kappa * (1.0 - e ** 2) / (2.0 * p)
                lam = sd * np.sqrt(kappa * p)
                M0 = M0_for(0.0, 0.0, eta, lam, om, kappa)
                uu, tt = orbit_state(np.array([dtau]), eta, lam, om, M0, kappa)
                if dist2(uu[0], tt[0], u1, dth) < 1e-10:
                    out.append((float(eta), float(lam), float(om)))
    return out


def solve_kepler(M, e, tol=1e-12, maxiter=60):
    M = wrap_pi(M)
    E = M + 0.85 * e * np.sign(np.sin(M))          # Danby (1987) starter
    for _ in range(maxiter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def true_from_ecc(E, e):
    return 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(0.5 * E), np.sqrt(1.0 - e) * np.cos(0.5 * E))


def mean_from_true(f, e):
    f = wrap_pi(f)
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(0.5 * f), np.sqrt(1.0 + e) * np.cos(0.5 * f))
    return E - e * np.sin(E)


def orbit_state(tau, eta, lam, theta0, M0, kappa):
    """u(tau), theta(tau).  theta = theta0 + sign(lambda) f, f = true anomaly,
    M = M0 + n tau.  Params may have shape (W, 1) against tau of shape (N,)."""
    p, e, _, n = orbit_elements(eta, lam, kappa)
    E = solve_kepler(M0 + n * tau, e)
    f = true_from_ecc(E, e)
    return p / (1.0 + e * np.cos(f)), theta0 + np.sign(lam) * f


def M0_for(tau_a, theta_a, eta, lam, theta0, kappa):
    """Mean anomaly at tau = 0 such that theta(tau_a) = theta_a."""
    _, e, _, n = orbit_elements(eta, lam, kappa)
    f_a = np.sign(lam) * (theta_a - theta0)
    return mean_from_true(f_a, e) - n * tau_a


def occluded(u1, th1, u2, th2):
    """True where the SEGMENT joining bodies 1 and 2 crosses the planet (u < 1).
    xi^2 = |r1 - r2|^2.  s* = (xi^2 + u2^2 - u1^2)/(2 xi^2) is the foot of the
    perpendicular from the origin on the line r2 + s (r1 - r2); x^2 = u2^2 - s*^2 xi^2
    is the notes' x^2.  Occlusion requires 0 < s* < 1 AND x^2 < 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        xi2 = u1 * u1 + u2 * u2 - 2.0 * u1 * u2 * np.cos(th1 - th2)
        s = (xi2 + u2 * u2 - u1 * u1) / (2.0 * xi2)
        x2 = u2 * u2 - s * s * xi2
        return (xi2 > 1e-14) & (s > 0.0) & (s < 1.0) & (x2 < 1.0)


def margin(u1, th1, u2, th2):
    """1 - d_seg^2, d_seg = distance from the planet centre to the segment joining the
    bodies.  Continuous in time; occluded <=> margin > 0 (both bodies have u >= 1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        xi2 = u1 * u1 + u2 * u2 - 2.0 * u1 * u2 * np.cos(th1 - th2)
        s = (xi2 + u2 * u2 - u1 * u1) / (2.0 * xi2)
        inside = (xi2 > 1e-14) & (s > 0.0) & (s < 1.0)
        d2 = np.where(inside, u2 * u2 - s * s * xi2, np.minimum(u1 * u1, u2 * u2))
    return 1.0 - d2


def dist2(u1, th1, u2, th2):
    return u1 * u1 + u2 * u2 - 2.0 * u1 * u2 * np.cos(th1 - th2)


def bool_runs(mask, tau):
    """Contiguous True runs of mask on a uniform midpoint grid -> [(a, b)]."""
    h = tau[1] - tau[0]
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    dm = np.diff(m)
    starts = np.flatnonzero(dm == 1)
    ends = np.flatnonzero(dm == -1)
    return [(max(tau[s] - h / 2, 0.0), min(tau[e - 1] + h / 2, 1.0)) for s, e in zip(starts, ends)]


@dataclass
class Orbit:
    name: str
    eta: float
    lam: float
    theta0: float
    M0: float
    kappa: float

    @property
    def el(self):
        return orbit_elements(self.eta, self.lam, self.kappa)

    def at(self, tau):
        return orbit_state(np.asarray(tau, float), self.eta, self.lam, self.theta0, self.M0, self.kappa)

    def xy(self, tau):
        u, th = self.at(tau)
        return u * np.cos(th), u * np.sin(th)

    def ellipse_xy(self, n=721):
        p, e, _, _ = self.el
        f = np.linspace(-np.pi, np.pi, n)
        u = p / (1.0 + e * np.cos(f))
        th = self.theta0 + np.sign(self.lam) * f
        return u * np.cos(th), u * np.sin(th)

    def describe(self, day):
        p, e, a, n = self.el
        T_h = day_length(day) / 3600.0
        P_tau = TWO_PI / n
        tau_peri = (-self.M0 / n) % P_tau
        return dict(e=e, a=a, p=p, q=p / (1 + e), Q=p / (1 - e), P_h=P_tau * T_h,
                    revs=1.0 / P_tau, sense="prograde" if self.lam > 0 else "retrograde",
                    t_peri=tau_clock(tau_peri, day))

    def values(self):
        return [self.eta, self.lam, self.theta0 % TWO_PI, self.M0]


def orbit_from_saved(name, s, kappa):
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise ValueError
    return Orbit(name, v[0], v[1], v[2], v[3], kappa)


def real_silences(vessel: Orbit, home: Orbit, n=20000):
    tau = (np.arange(n) + 0.5) / n
    u, th = vessel.at(tau)
    ub, thb = home.at(tau)
    return bool_runs(occluded(u, th, ub, thb), tau)


# =============================================================================
# MC1: vessel + home ship + kappa + L
# =============================================================================
class MC1:
    names = ("eta", "lambda", "theta0", "eta_b", "lambda_b", "theta0_b", "kappa", "L")
    ndim = 8

    def __init__(self, T_s, sil_tau, cfg, n_grid=None):
        self.T = T_s
        self.cfg = cfg
        self.kmin = kappa_from_rho(cfg["rho_min"], T_s)
        self.kmax = kappa_from_rho(cfg["rho_max"], T_s)
        N = int(n_grid or cfg["n_grid"])
        self.N = N
        self.tau_all = np.linspace(0.0, 1.0, N + 1)           # cell edges
        # input silence inside each cell, as [ia, ib] in cell units (one interval per cell)
        self.ia = np.zeros(N)
        self.ib = np.zeros(N)
        left = self.tau_all[:-1]
        for a, b in sil_tau:
            lo = np.clip((a - left) * N, 0.0, 1.0)
            hi = np.clip((b - left) * N, 0.0, 1.0)
            put = hi > lo
            self.ia[put], self.ib[put] = lo[put], hi[put]
        self._sil = list(sil_tau)
        self.beta = 1.0

    # ------------------------------------------------ posterior in the notes' parameters Y
    def log_prior(self, Y):
        eta, lam, th0, etab, lamb, th0b, kap, L = Y.T
        c = self.cfg
        with np.errstate(all="ignore"):
            ok = (kap >= self.kmin) & (kap <= self.kmax) & (L >= 1.0)
            ok &= orbit_ok(eta, lam, kap, L) & orbit_ok(etab, lamb, kap, L)
            if not c["allow_retrograde"]:
                ok &= (lam > 0) & (lamb > 0)
            lp = -np.log(kap) - c["c1"] * L        # f: log-uniform rho;  g: exp(-c1 L)
            if c["f_is_kappa_marginal"]:
                lp = lp - 3.0 * np.log(kap)
        return np.where(ok, lp, -np.inf)

    def log_post(self, Y):
        lp = self.log_prior(Y)
        ok = np.isfinite(lp)
        if ok.any():
            lp[ok] = lp[ok] + self.log_like(Y[ok])
        return lp

    # ------------------------------------------------ sampling coordinates X
    # X = (s1, w, theta0, s1_b, w_b, theta0_b, log kappa, L): a box (see geo_to_phys)
    def x_to_Y(self, X):
        s1, w, th0, s1b, wb, th0b, lk, L = X.T
        c = self.cfg
        kap = np.exp(lk)
        box = ((s1 >= 0) & (s1 <= 1) & (np.abs(w) <= 1) & (w != 0)
               & (s1b >= 0) & (s1b <= 1) & (np.abs(wb) <= 1) & (wb != 0)
               & (lk >= np.log(self.kmin)) & (lk <= np.log(self.kmax)) & (L > 1.0)
               & (th0 >= -TWO_PI) & (th0 <= 2 * TWO_PI) & (th0b >= -TWO_PI) & (th0b <= 2 * TWO_PI))
        if not c["allow_retrograde"]:
            box &= (w > 0) & (wb > 0)
        eta, lam, j1 = geo_to_phys(s1, w, kap, L)
        etab, lamb, j2 = geo_to_phys(s1b, wb, kap, L)
        Y = np.stack([eta, lam, th0, etab, lamb, th0b, kap, L], axis=1)
        logj = j1 + j2 + lk                                    # d kappa = kappa d(log kappa)
        return Y, logj, box & np.isfinite(logj)

    def terms(self, Y):
        """Likelihood ingredients for each row of Y (all rows must be prior-valid)."""
        eta, lam, th0, etab, lamb, th0b, kap, L = [c[:, None] for c in Y.T]
        M0 = M0_for(0.0, 0.0, eta, lam, th0, kap)
        M0b = M0_for(0.0, 0.0, etab, lamb, th0b, kap)
        u, th = orbit_state(self.tau_all, eta, lam, th0, M0, kap)
        ub, thb = orbit_state(self.tau_all, etab, lamb, th0b, M0b, kap)
        # (1/T) int Theta dt = |real XOR input| / T, with the real silence obtained from the
        # margin linearly interpolated inside each cell (continuous in the parameters)
        m = margin(u, th, ub, thb)
        m0, m1 = m[:, :-1], m[:, 1:]
        den = m0 - m1
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.clip(np.where(den != 0.0, m0 / den, 0.5), 0.0, 1.0)
        ra = np.where(m0 > 0.0, 0.0, c)
        rb = np.where(m1 > 0.0, 1.0, c)
        inter = np.clip(np.minimum(rb, self.ib) - np.maximum(ra, self.ia), 0.0, None)
        mism = np.sum((rb - ra) + (self.ib - self.ia) - 2.0 * inter, axis=1) / self.N
        xi0 = dist2(u[:, 0], th[:, 0], ub[:, 0], thb[:, 0])
        xi1 = dist2(u[:, -1], th[:, -1], ub[:, -1], thb[:, -1])
        return mism, xi0, xi1

    def log_like(self, Y):
        c = self.cfg
        mism, xi0, xi1 = self.terms(Y)
        return (-0.5 * (mism / c["sigma_tau"]) ** 2
                - 0.5 * xi0 / c["sigma_xi0"] ** 2
                - 0.5 * xi1 / c["sigma_xi1"] ** 2)

    def __call__(self, X):
        """Tempered log target in X: log pi(Y) + log|J| + beta log L(X|Y)."""
        X = np.atleast_2d(X)
        Y, logj, ok = self.x_to_Y(X)
        out = np.full(len(X), -np.inf)
        if ok.any():
            lp = self.log_prior(Y[ok]) + logj[ok]
            fin = np.isfinite(lp)
            idx = np.flatnonzero(ok)[fin]
            if idx.size:
                out[idx] = lp[fin] + self.beta * self.log_like(Y[idx])
        out[np.isnan(out)] = -np.inf
        return out

    def y_to_x(self, Y):
        eta, lam, th0, etab, lamb, th0b, kap, L = Y.T
        s1, w = phys_to_geo(eta, lam, kap, L)
        s1b, wb = phys_to_geo(etab, lamb, kap, L)
        return np.stack([s1, w, th0 % TWO_PI, s1b, wb, th0b % TWO_PI, np.log(kap), L], axis=1)

    def x_span(self):
        return np.array([1, 1, TWO_PI, 1, 1, TWO_PI, np.log(self.kmax / self.kmin), 2.0])

    def search(self, rng):
        """Screen vessel orbits; for each, every home-ship orbit that docks exactly at
        tau = 0 and tau = 1 (Lambert).  Returns candidate Y rows and log posteriors."""
        c = self.cfg
        lk0, lk1 = np.log(self.kmin), np.log(self.kmax)
        screen = MC1(self.T, self._sil, c, n_grid=c["n_grid_screen"])
        span = np.array([1.0, 1.0, TWO_PI, lk1 - lk0, 2.0])

        def draw(n):
            sg = rng.choice([-1.0, 1.0], n) if c["allow_retrograde"] else np.ones(n)
            return np.stack([rng.uniform(0, 1, n), sg * rng.uniform(0.02, 1, n), rng.uniform(0, TWO_PI, n),
                             rng.uniform(lk0, lk1, n), np.minimum(1.2 + rng.exponential(1.0 / c["c1"], n), 20.0)],
                            axis=1)

        def around(V, n, scale):
            base = V[rng.integers(len(V), size=n)]
            X = base + rng.normal(size=base.shape) * span * scale
            X[:, 0] = np.clip(X[:, 0], 0.0, 1.0)
            X[:, 1] = np.sign(base[:, 1]) * np.clip(np.abs(X[:, 1]), 1e-3, 1.0)
            X[:, 3] = np.clip(X[:, 3], lk0, lk1)
            X[:, 4] = np.clip(X[:, 4], 1.05, 20.0)
            return X

        def evaluate(V):
            rows, owner = [], []
            for k, (s1, w, th0, lk, L) in enumerate(V):
                kap = float(np.exp(lk))
                eta, lam, _ = geo_to_phys(s1, w, kap, L)
                eta, lam = float(eta), float(lam)
                v = Orbit("v", eta, lam, th0, float(M0_for(0.0, 0.0, eta, lam, th0, kap)), kap)
                u0 = float(v.at(0.0)[0])
                u1, th1 = (float(x) for x in v.at(1.0))
                for eb, lb, ob in lambert_2d(u0, u1, th1, 1.0, kap, L, c["allow_retrograde"]):
                    rows.append([eta, lam, th0, eb, lb, ob, kap, L])
                    owner.append(k)
            if not rows:
                return np.empty((0, 8)), np.empty(0), np.empty(0, int)
            Y = np.array(rows)
            lp = np.concatenate([screen.log_post(Y[i:i + 2000]) for i in range(0, len(Y), 2000)])
            return Y, lp, np.array(owner)

        allY = [np.empty((0, 8))]
        Vs, Vlp = np.empty((0, 5)), np.empty(0)          # every vessel tried, best lp of its pairs
        t0 = time.time()
        for r, (n, scale) in enumerate(c["search_rounds"]):
            if scale is None or not np.isfinite(Vlp).any():
                V = draw(n)
            else:
                top = np.argsort(Vlp)[::-1][:c["search_top"]]
                V = around(Vs[top], n, scale)
            Y, lp, owner = evaluate(V)
            allY.append(Y)
            best = np.full(len(V), -np.inf)
            if len(lp):
                np.maximum.at(best, owner, lp)
            Vs, Vlp = np.concatenate([Vs, V]), np.concatenate([Vlp, best])
            if len(lp):
                mism, _, _ = self.terms(Y[np.argmax(lp)][None])
                print(f"  MC1 search round {r + 1}: {n} vessel orbits, {len(lp)} docking home-ship orbits, "
                      f"best log p = {lp.max():.2f}, silence mismatch = {mism[0] * self.T / 60:.1f} min "
                      f"({time.time() - t0:.0f} s)")
            else:
                print(f"  MC1 search round {r + 1}: no docking solutions")
        Y = np.concatenate(allY)
        return Y, chunked(self.log_post, Y)

    def orbits(self, y):
        """Orbits of sample y.  L is returned at its conditional mode max(Q, Q_b):
        L enters only pi(Y) via exp(-c1 L) on L >= max(Q, Q_b), so this raises the posterior."""
        eta, lam, th0, etab, lamb, th0b, kap, L = y
        v = Orbit("Vessel", eta, lam, th0, float(M0_for(0.0, 0.0, eta, lam, th0, kap)), kap)
        h = Orbit("Home ship", etab, lamb, th0b, float(M0_for(0.0, 0.0, etab, lamb, th0b, kap)), kap)
        L_star = max(v.describe((0, 1))["Q"], h.describe((0, 1))["Q"], 1.0)
        return v, h, kap, L_star


# =============================================================================
# MC2: junky station
# =============================================================================
class MC2:
    names = ("eta_js", "lambda_js", "theta0_js")
    ndim = 3

    def __init__(self, vessel: Orbit, kappa, L, tau_b, tau_e, cfg):
        self.kappa, self.L, self.cfg = kappa, L, cfg
        self.vessel = vessel
        self.tau_b, self.tau_e = tau_b, tau_e
        self.ub, self.thb = [float(x) for x in vessel.at(tau_b)]
        self.ue, self.the = [float(x) for x in vessel.at(tau_e)]
        self.beta = 1.0

    def log_prior(self, Y):
        eta, lam, th0 = Y.T
        with np.errstate(all="ignore"):
            ok = orbit_ok(eta, lam, self.kappa, self.L)
            ok &= (th0 >= -TWO_PI) & (th0 <= 2 * TWO_PI)
            if not self.cfg["allow_retrograde"]:
                ok &= lam > 0
            _, e, _, _ = orbit_elements(eta, lam, self.kappa)
            lp = np.log1p(-e ** 2)                         # (1 - e_js^2)
        return np.where(ok, lp, -np.inf)

    def anchored(self, Y):
        """Station state anchored at theta_js(tau_b) = theta_vessel(tau_b)."""
        eta, lam, th0 = Y.T
        p, e, _, n = orbit_elements(eta, lam, self.kappa)
        s = np.sign(lam)
        f_b = wrap_pi(s * (self.thb - th0))
        M_b = mean_from_true(f_b, e)
        u_b = p / (1.0 + e * np.cos(f_b))
        f_e = true_from_ecc(solve_kepler(M_b + n * (self.tau_e - self.tau_b), e), e)
        u_e, th_e = p / (1.0 + e * np.cos(f_e)), th0 + s * f_e
        return u_b, u_e, th_e, M_b - n * self.tau_b

    def log_like(self, Y):
        u_b, u_e, th_e, _ = self.anchored(Y)
        xb = (u_b - self.ub) ** 2
        xe = dist2(u_e, th_e, self.ue, self.the)
        return -0.5 * (xb + xe) / self.cfg["sigma_js"] ** 2

    def log_post(self, Y):
        lp = self.log_prior(Y)
        ok = np.isfinite(lp)
        if ok.any():
            lp[ok] = lp[ok] + self.log_like(Y[ok])
        return lp

    # X = (s1, w, theta0): a box (see geo_to_phys); kappa and L fixed by MC1
    def x_to_Y(self, X):
        s1, w, th0 = X.T
        box = ((s1 >= 0) & (s1 <= 1) & (np.abs(w) <= 1) & (w != 0)
               & (th0 >= -TWO_PI) & (th0 <= 2 * TWO_PI))
        if not self.cfg["allow_retrograde"]:
            box &= w > 0
        eta, lam, logj = geo_to_phys(s1, w, self.kappa, self.L)
        return np.stack([eta, lam, th0], axis=1), logj, box & np.isfinite(logj)

    def __call__(self, X):
        X = np.atleast_2d(X)
        Y, logj, ok = self.x_to_Y(X)
        out = np.full(len(X), -np.inf)
        if ok.any():
            lp = self.log_prior(Y[ok]) + logj[ok]
            fin = np.isfinite(lp)
            idx = np.flatnonzero(ok)[fin]
            if idx.size:
                out[idx] = lp[fin] + self.beta * self.log_like(Y[idx])
        out[np.isnan(out)] = -np.inf
        return out

    def y_to_x(self, Y):
        eta, lam, th0 = Y.T
        s1, w = phys_to_geo(eta, lam, self.kappa, self.L)
        return np.stack([s1, w, th0 % TWO_PI], axis=1)

    def x_span(self):
        return np.array([1.0, 1.0, TWO_PI])

    def search(self, rng=None):
        """Every station orbit through the vessel's positions at tau_b and tau_e (Lambert)."""
        sols = lambert_2d(self.ub, self.ue, self.the - self.thb, self.tau_e - self.tau_b,
                          self.kappa, self.L, self.cfg["allow_retrograde"])
        v = self.vessel
        # the vessel's own orbit is a trivial solution (station flying in formation): exclude it
        sols = [s for s in sols if not (abs(s[0] - v.eta) < 1e-6 * abs(v.eta) and abs(s[1] - v.lam) < 1e-6 * abs(v.lam))]
        Y = np.array([[eta, lam, om + self.thb] for eta, lam, om in sols]) if sols else np.empty((0, 3))
        print(f"  MC2 search: {len(Y)} station orbits (other than the vessel's own) pass through both encounter points")
        return Y, chunked(self.log_post, Y)

    def orbit(self, y):
        _, _, _, M0 = self.anchored(np.atleast_2d(y))
        return Orbit("Junky station", y[0], y[1], y[2], float(M0[0]), self.kappa)


# =============================================================================
# SAMPLER DRIVER
# =============================================================================
def chunked(fn, Y, size=2000):
    if len(Y) == 0:
        return np.empty(0)
    return np.concatenate([fn(Y[i:i + size]) for i in range(0, len(Y), size)])


def pick_modes(model, Y, lp, n_modes, min_sep=0.02):
    """Greedy: best candidates first, skipping any within min_sep (box units) of one kept."""
    keep = []
    if len(Y) == 0:
        return np.empty((0, model.ndim))
    X = model.y_to_x(Y) / model.x_span()
    for i in np.argsort(lp)[::-1]:
        if not np.isfinite(lp[i]):
            break
        d = X[i]
        if all(np.max(np.abs(np.where(np.arange(len(d)) % 3 == 2,
                                       (d - X[j] + 0.5) % 1.0 - 0.5, d - X[j]))) > min_sep for j in keep):
            keep.append(i)
        if len(keep) >= n_modes:
            break
    return Y[keep]


def refine_map(model, Y_modes, label):
    """Nelder-Mead on -log posterior (notes' parameters) from each candidate, in box
    coordinates with a small initial simplex.  Returns refined Y and log posteriors."""
    from scipy.optimize import minimize
    span = model.x_span()

    def neg_log_post(x):
        Y, _, ok = model.x_to_Y(x[None])
        if not ok[0]:
            return 1e300
        v = model.log_post(Y)[0]
        return -v if np.isfinite(v) else 1e300

    Ys, lps = [], []
    t0 = time.time()
    for y in Y_modes:
        x0 = model.y_to_x(y[None])[0]
        simplex = np.vstack([x0] + [x0 + np.eye(x0.size)[i] * span[i] * 2e-3 for i in range(x0.size)])
        f0 = neg_log_post(x0)
        res = minimize(neg_log_post, x0, method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, maxfev=400 * x0.size, xatol=1e-9,
                                    fatol=1e-6, adaptive=True))
        x = res.x if res.fun < f0 else x0
        Ys.append(model.x_to_Y(x[None])[0][0])
        lps.append(-min(res.fun, f0))
    if len(Ys):
        print(f"  {label} MAP refinement of {len(Ys)} candidates: best log p = {max(lps):.2f} "
              f"({time.time() - t0:.0f} s)")
    return np.array(Ys).reshape(-1, model.ndim), np.array(lps)


def run_modes(model, Y_modes, walkers, steps, rng, label):
    """A small emcee ensemble on the notes' posterior around each mode.  The sampled
    density is pi(Y) L(X|Y) expressed in the box coordinates (Jacobian included)."""
    import emcee
    span = model.x_span()
    out = [Y_modes]
    t0 = time.time()
    for k, y in enumerate(Y_modes):
        x0 = model.y_to_x(y[None])[0]
        P = []
        tries = 0
        while len(P) < walkers and tries < 50 * walkers:
            tries += 1
            x = x0 + rng.normal(size=x0.size) * span * 1e-5
            if np.isfinite(model(x[None])[0]):
                P.append(x)
        if len(P) < walkers:
            continue
        s = emcee.EnsembleSampler(walkers, model.ndim, model, vectorize=True,
                                  moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
        s.run_mcmc(np.array(P), steps, progress=False, skip_initial_state_check=True)
        X = s.get_chain(discard=steps // 2, flat=True)
        Y, _, ok = model.x_to_Y(X)
        out.append(Y[ok])
        print(f"  {label} emcee mode {k + 1}/{len(Y_modes)}: log p (start) = {model.log_post(y[None])[0]:.2f}, "
              f"acceptance {np.mean(s.acceptance_fraction):.2f} ({time.time() - t0:.0f} s)")
    Y = np.unique(np.concatenate(out), axis=0)
    lp = chunked(model.log_post, Y)
    good = np.isfinite(lp)
    Y, lp = Y[good], lp[good]
    order = np.argsort(lp)
    return Y[order], lp[order]


def thin_for_slider(chain, logp, n_show):
    n = len(logp)
    if n <= n_show:
        return chain, logp
    idx = np.unique(np.round(np.linspace(0, n - 1, n_show)).astype(int))
    return chain[idx], logp[idx]


def cached_run(cache: Path, key: str, runner):
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if str(z["key"]) == key:
                print(f"  using cached samples {cache.name}")
                return z["chain"], z["logp"]
        except Exception:
            pass
    chain, logp = runner()
    try:
        np.savez(cache, chain=chain, logp=logp, key=np.array(key))
    except OSError as exc:
        print(f"  (could not write cache: {exc})")
    return chain, logp


# =============================================================================
# SAVED PARAMETERS: read and write
# =============================================================================
def fmt_vals(vals):
    return ", ".join(f"{float(v):.17g}" for v in vals)


def fmt_interval(a, b):
    return f"[{s_to_clock(a)} - {s_to_clock(b)}]"


def fmt_runs(runs, day):
    return ", ".join(f"[{tau_clock(a, day)} - {tau_clock(b, day)}]" for a, b in runs) or "none"


def saved_mc1(saved):
    try:
        iv = find_intervals(saved["MC1_DAY"])
        day = (iv[0][0], iv[0][1])
        kappa = float(saved["MC1_KAPPA"])
        L = float(saved["MC1_L"])
        v = orbit_from_saved("Vessel", saved["MC1_VESSEL"], kappa)
        h = orbit_from_saved("Home ship", saved["MC1_HOMESHIP"], kappa)
        return dict(day=day, kappa=kappa, L=L, vessel=v, home=h, model=saved.get("MC1_MODEL", ""))
    except (KeyError, ValueError, IndexError, TypeError):
        return None


def saved_mc2(saved, kappa):
    try:
        js = orbit_from_saved(saved.get("MC2_JS_NAME", "Junky station"), saved["MC2_JS"], kappa)
        iv = find_intervals(saved.get("MC2_JS_INTERVAL", ""))
        interval = (iv[0][0], iv[0][1]) if iv and iv[0][1] is not None else None
        return dict(js=js, name=saved.get("MC2_JS_NAME", "Junky station"), interval=interval,
                    model=saved.get("MC2_MODEL", ""))
    except (KeyError, ValueError, IndexError, TypeError):
        return None


def js_encounters(js: Orbit, vessel: Orbit, interval, day):
    """Real encounter times: minimum vessel-station distance near each input time."""
    out = []
    for c in interval:
        tc = clock_tau(c, day)
        tau = np.clip(np.linspace(tc - 0.02, tc + 0.02, 4001), 0.0, 1.0)
        u, th = vessel.at(tau)
        uj, thj = js.at(tau)
        d2 = dist2(u, th, uj, thj)
        i = int(np.argmin(d2))
        out.append((tau[i], float(np.sqrt(max(d2[i], 0.0)))))
    return out


def saved_block_lines(mc1, mc2, extra):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = ["SAVED PARAMETERS",
         f"# written by orbital.py v0.1 on {now}. Delete a MC1_ line to rerun MC1, a MC2_ line to rerun MC2.",
         "# orbit lines: eta, lambda, theta0 [rad], M0 (mean anomaly at tau = 0) [rad]"]
    if mc1:
        day, kap = mc1["day"], mc1["kappa"]
        v, h = mc1["vessel"], mc1["home"]
        T = day_length(day)
        L += [f"MC1_DAY = {fmt_interval(*day)}",
              f"MC1_KAPPA = {kap:.17g}",
              f"MC1_L = {mc1['L']:.17g}",
              f"MC1_VESSEL = {fmt_vals(v.values())}",
              f"MC1_HOMESHIP = {fmt_vals(h.values())}",
              f"MC1_MODEL = {mc1['model']}"]
        rho = rho_from_kappa(kap, T)
        P_surf = TWO_PI / np.sqrt(kap) * T / 3600.0
        L.append(f"# planet: rho = {rho:.1f} kg/m^3, circular period at the surface = {P_surf:.3f} h, T = {T / 3600:.4g} h")
        for o in (v, h):
            dsc = o.describe(day)
            L.append(f"# {o.name}: e = {dsc['e']:.4f}, a = {dsc['a']:.4f} R, peri = {dsc['q']:.4f} R, "
                     f"apo = {dsc['Q']:.4f} R, P = {dsc['P_h']:.4f} h, {dsc['revs']:.3f} rev/day, "
                     f"{dsc['sense']}, first pericentre {dsc['t_peri']}")
        runs = real_silences(v, h)
        u0, th0 = v.at(0.0)
        ub0, thb0 = h.at(0.0)
        u1, th1 = v.at(1.0)
        ub1, thb1 = h.at(1.0)
        L.append(f"# real silences: {fmt_runs(runs, day)}")
        L.append(f"# xi(0) = {np.sqrt(dist2(u0, th0, ub0, thb0)):.4g} R, xi(1) = {np.sqrt(dist2(u1, th1, ub1, thb1)):.4g} R")
    if mc2:
        js = mc2["js"]
        L += [f"MC2_JS_NAME = {mc2['name']}",
              f"MC2_JS_INTERVAL = {fmt_interval(*mc2['interval'])}",
              f"MC2_JS = {fmt_vals(js.values())}",
              f"MC2_MODEL = {mc2['model']}"]
        if mc1:
            dsc = js.describe(mc1["day"])
            enc = js_encounters(js, mc1["vessel"], mc2["interval"], mc1["day"])
            L.append(f"# {mc2['name']}: e = {dsc['e']:.4f}, a = {dsc['a']:.4f} R, P = {dsc['P_h']:.4f} h, {dsc['sense']}")
            L.append("# real encounters: " + ", ".join(
                f"{tau_clock(t, mc1['day'])} (distance {d:.3g} R)" for t, d in enc))
    for k, v in extra.items():
        L.append(f"{k} = {v}")
    L.append("END SAVED PARAMETERS")
    return L


PROGRAM_KEYS = ("MC1_DAY", "MC1_KAPPA", "MC1_L", "MC1_VESSEL", "MC1_HOMESHIP", "MC1_MODEL",
                "MC2_JS_NAME", "MC2_JS_INTERVAL", "MC2_JS", "MC2_MODEL")


def read_text(path: Path):
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8"


def write_saved(path: Path, mc1, mc2, saved_old):
    text, enc = read_text(path)
    extra = {k: v for k, v in saved_old.items() if k not in PROGRAM_KEYS}
    block = saved_block_lines(mc1, mc2, extra)
    lines = text.splitlines()
    i0 = next((i for i, l in enumerate(lines) if l.strip().startswith("SAVED PARAMETERS")), None)
    i1 = None
    if i0 is not None:
        i1 = next((i for i in range(i0 + 1, len(lines))
                   if lines[i].strip().startswith("END SAVED PARAMETERS")), None)
    if i0 is not None and i1 is not None:
        new = lines[:i0] + block + lines[i1 + 1:]
    else:
        new = block + [""] + lines
    shutil.copy2(path, path.with_name(path.name + ".bak"))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(new) + "\n", encoding="utf-8" if enc == "utf-8-sig" else enc)
    os.replace(tmp, path)
    print(f"  SAVED PARAMETERS written into {path.name} (previous version: {path.name}.bak)")
    print("\n".join("    " + l for l in block))


# =============================================================================
# REMINDERS
# =============================================================================
def stable_seed(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def reminder_schedule(r: Reminder, d: DayFile, day):
    """[(offset_s from day start, message)]; deterministic for a given file content."""
    if not r.messages:
        return []
    rng = np.random.default_rng(stable_seed(d.date, r.name, r.start, r.end, r.mu, r.sd, *r.messages))
    o = clock_offset(r.start, day)
    o_end = o + ((r.end - r.start) % 86400)
    out = []
    while o <= o_end + 1e-9:
        out.append((o, r.messages[int(rng.integers(len(r.messages)))], r))
        o += max(rng.normal(r.mu, r.sd), 0.5) * 60.0
    return out


def speak(text, voice, enabled):
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] REMINDER: {text}")
    if not enabled or os.name != "nt":
        return
    t = text.replace("'", "''")
    v = voice.replace("'", "''")
    ps = ("Add-Type -AssemblyName System.Speech; "
          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$v = $s.GetInstalledVoices() | ForEach-Object {{ $_.VoiceInfo.Name }} | "
          f"Where-Object {{ $_ -like '*{v}*' }} | Select-Object -First 1; "
          "if ($v) { $s.SelectVoice($v) }; "
          f"$s.Speak('{t}')")
    try:
        subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as exc:
        print(f"  (voice failed: {exc})")


# =============================================================================
# EARNINGS
# =============================================================================
def earnings(d: DayFile):
    rows, total, avail_total = [], Counter(), Counter()
    for b in d.blocks:
        got, avail = Counter(), Counter()
        for e in b.emeralds:
            avail.update(e.prizes)
            if e.done:
                got.update(e.prizes)
        done = sum(e.done for e in b.emeralds)
        got["EMERALD"] = done
        avail["EMERALD"] = len(b.emeralds)
        total.update(got)
        avail_total.update(avail)
        rows.append((b, got, avail))
    return rows, total, avail_total


SHORT = {"EMERALD": "E", "SAPPHIRE": "S", "DIAMOND": "D", "BALL": "B", "MARTIAN": "M"}


def fmt_earn(got, avail, emoji=True):
    parts = []
    for k in KNOWN_PRIZES:
        if avail.get(k, 0) or got.get(k, 0):
            tag = EMOJI[k] if emoji else SHORT[k]
            parts.append(f"{tag} {got.get(k, 0)}/{avail.get(k, 0)}")
    return "  ".join(parts) if parts else "-"


# =============================================================================
# PLOTTING
# =============================================================================
def time_cmap():
    from matplotlib.colors import ListedColormap, hsv_to_rgb
    h = np.linspace(0.0, 1.0, 256, endpoint=False)
    hsv = np.stack([(2.0 / 3.0 + h) % 1.0, np.full_like(h, 0.85), np.full_like(h, 0.95)], axis=1)
    return ListedColormap(hsv_to_rgb(hsv), name="localtime_0h_blue")


def local_hours(tau, day):
    return ((day[0] + np.asarray(tau) * day_length(day)) / 3600.0) % 24.0


def make_figure(title):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig = plt.figure(figsize=(15, 8.4))
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    ax_orb = fig.add_axes([0.02, 0.13, 0.50, 0.82])
    ax_cb = fig.add_axes([0.525, 0.18, 0.010, 0.72])
    ax_tl = fig.add_axes([0.60, 0.70, 0.38, 0.24])
    ax_tx = fig.add_axes([0.60, 0.10, 0.38, 0.55])
    sm = ScalarMappable(norm=Normalize(0, 24), cmap=time_cmap())
    cb = fig.colorbar(sm, cax=ax_cb, ticks=range(0, 25, 3))
    cb.set_label("local time [h]")
    return fig, dict(orb=ax_orb, tl=ax_tl, tx=ax_tx)


def colored_path(ax, orbit, day, lw, n=2400, zorder=3, alpha=1.0):
    from matplotlib.collections import LineCollection
    tau = np.linspace(0.0, 1.0, n)
    x, y = orbit.xy(tau)
    pts = np.stack([x, y], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    hrs = local_hours(0.5 * (tau[:-1] + tau[1:]), day)
    lc = LineCollection(segs, cmap=time_cmap(), linewidths=lw, zorder=zorder, alpha=alpha)
    lc.set_array(hrs)
    lc.set_clim(0, 24)
    ax.add_collection(lc)


def render(axes, sc):
    """sc: scene dict (see build_scene_*)."""
    from matplotlib.patches import Circle
    day = sc["day"]
    ax = axes["orb"]
    ax.clear()
    L = sc["L"]
    ax.add_patch(Circle((0, 0), 1.0, color="0.35", zorder=2))
    ax.add_patch(Circle((0, 0), L, fill=False, ls=":", lw=0.8, color="0.6", zorder=1))
    for orb, style in sc["orbits"]:
        ex, ey = orb.ellipse_xy()
        ax.plot(ex, ey, color="0.75", lw=0.7, zorder=1)
        colored_path(ax, orb, day, style.get("lw", 1.5))
    vessel = sc["orbits"][0][0]
    home = sc["orbits"][1][0] if len(sc["orbits"]) > 1 else None
    for a, b in sc.get("shadows_real", []):
        tau = np.linspace(a, b, 300)
        x, y = vessel.xy(tau)
        ax.plot(x, y, color="k", lw=7, alpha=0.25, zorder=2.5, solid_capstyle="butt")
        if home is not None:
            x, y = home.xy(tau)
            ax.plot(x, y, color="k", lw=5, alpha=0.18, zorder=2.5, solid_capstyle="butt")
    # launch / return
    x0, y0 = vessel.xy(0.0)
    ax.plot(x0, y0, "o", ms=9, mfc="white", mec="k", zorder=6)
    x1, y1 = vessel.xy(1.0)
    ax.plot(x1, y1, "X", ms=9, mfc="white", mec="k", zorder=6)
    if home is not None:
        xb1, yb1 = home.xy(1.0)
        ax.plot(xb1, yb1, "+", ms=12, mew=2, color="k", zorder=6)
    # hour ticks on the vessel path
    T_h = day_length(day) / 3600.0
    h0 = day[0] / 3600.0
    for k in range(int(np.ceil(h0)), int(np.floor(h0 + T_h)) + 1):
        t = (k - h0) / T_h
        x, y = vessel.xy(t)
        ax.plot(x, y, ".", color="k", ms=4, zorder=5)
        ax.annotate(f"{k % 24}h", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7, color="0.2")
    for t, lab, mk, col in sc.get("markers", []):
        o = vessel
        if isinstance(t, tuple):
            o, t = t
        x, y = o.xy(t)
        ax.plot(x, y, mk, color=col, ms=8, mec="k", mew=0.6, zorder=7)
        if lab:
            ax.annotate(lab, (x, y), xytext=(5, -9), textcoords="offset points", fontsize=7.5, color=col)
    for t in sc.get("reminders", []):
        x, y = vessel.xy(t)
        ax.plot(x, y, ".", color="tab:purple", ms=3, alpha=0.6, zorder=4)
    if sc.get("now") is not None:
        tn = sc["now"]
        for orb, style in sc["orbits"]:
            x, y = orb.xy(tn)
            ax.plot(x, y, "o", ms=11, mfc=style.get("now_color", "w"), mec="k", mew=1.5, zorder=8)
            ax.annotate(orb.name, (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9,
                        fontweight="bold", zorder=9)
    lim = 1.05 * L
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(sc["title"], fontsize=11)
    # ------------------------------------------------------------------ timeline
    tl = axes["tl"]
    tl.clear()
    h0, h1 = day[0] / 3600.0, day[0] / 3600.0 + T_h

    def bars(runs, y, color, label):
        if runs:
            tl.broken_barh([(h0 + a * T_h, (b - a) * T_h) for a, b in runs], (y - 0.35, 0.7),
                           color=color, label=label)

    bars(sc.get("shadows_input", []), 3, "0.7", "input silence")
    bars(sc.get("shadows_real", []), 2, "k", "real silence")
    for name, a, b, kind in sc.get("blocks", []):
        col = "tab:green" if kind == "JUNKY STATION" else "tab:blue"
        tl.broken_barh([(h0 + a * T_h, (b - a) * T_h)], (0.65, 0.7), color=col, alpha=0.5)
        tl.text(h0 + 0.5 * (a + b) * T_h, 1.0, name, ha="center", va="center", fontsize=6.5, clip_on=True)
    for t, lab, mk, col in sc.get("markers", []):
        if isinstance(t, tuple):
            t = t[1]
        tl.plot(h0 + t * T_h, 0.0, mk, color=col, ms=6, mec="k", mew=0.5)
    for t in sc.get("reminders", []):
        tl.plot([h0 + t * T_h] * 2, [-0.3, -0.1], color="tab:purple", lw=0.6)
    if sc.get("now") is not None:
        tl.axvline(h0 + sc["now"] * T_h, color="r", lw=1.2)
    tl.set_xlim(h0, h1)
    tl.set_ylim(-0.6, 3.6)
    tl.set_yticks([0, 1, 2, 3])
    tl.set_yticklabels(["events", "blocks", "real", "input"], fontsize=8)
    ticks = np.arange(np.ceil(h0), np.floor(h1) + 1)
    tl.set_xticks(ticks)
    tl.set_xticklabels([f"{int(t) % 24}" for t in ticks], fontsize=8)
    tl.set_xlabel("local time [h]", fontsize=8)
    tl.grid(axis="x", alpha=0.3)
    # ------------------------------------------------------------------ text
    tx = axes["tx"]
    tx.clear()
    tx.axis("off")
    tx.text(0.0, 1.0, sc["info"], va="top", ha="left", family="monospace", fontsize=8.3,
            transform=tx.transAxes)


def orbit_table(orbs, day):
    names = [o.name[:13] for o in orbs]
    ds = [o.describe(day) for o in orbs]
    w = 14
    s = " " * 16 + "".join(f"{n[:w]:>{w}}" for n in names) + "\n"
    rows = [("e", "e", "{:.4f}"), ("a [R]", "a", "{:.3f}"), ("peri [R]", "q", "{:.3f}"),
            ("apo [R]", "Q", "{:.3f}"), ("P [h]", "P_h", "{:.3f}"), ("rev/day", "revs", "{:.3f}")]
    for lab, k, f in rows:
        s += f"{lab:<16}" + "".join(f"{f.format(d[k]):>{w}}" for d in ds) + "\n"
    s += f"{'sense':<16}" + "".join(f"{d['sense']:>{w}}" for d in ds) + "\n"
    s += f"{'1st pericentre':<16}" + "".join(f"{d['t_peri']:>{w}}" for d in ds) + "\n"
    return s


def planet_line(kappa, L, day):
    T = day_length(day)
    rho = rho_from_kappa(kappa, T)
    txt = f"kappa = {kappa:.2f}   rho = {rho:.0f} kg/m^3   L = {L:.3f}   T = {T / 3600:.3f} h"
    if CONFIG["R_km"]:
        txt += f"   R = {CONFIG['R_km']:g} km"
    return txt


# =============================================================================
# SELECTION WINDOW
# =============================================================================
def choose(chain, logp, build_scene, title, auto=False):
    n = len(logp)
    if n == 0:
        print("  no valid samples")
        return None
    if auto:
        return n - 1
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
    fig, axes = make_figure(title)
    ax_sl = fig.add_axes([0.07, 0.04, 0.45, 0.03])
    ax_bt = fig.add_axes([0.80, 0.025, 0.15, 0.055])
    state = {"saved": False}

    def redraw(v):
        i = int(v)
        render(axes, build_scene(chain[i], logp[i], i, n))
        fig.canvas.draw_idle()

    sl = Slider(ax_sl, "model (low -> high log p)", 0, n - 1, valinit=n - 1, valstep=1)
    sl.on_changed(redraw)
    bt = Button(ax_bt, "Save this model")

    def on_save(_):
        state["saved"] = True
        plt.close(fig)

    def on_key(ev):
        if ev.key in ("left", "right"):
            sl.set_val(min(max(int(sl.val) + (1 if ev.key == "right" else -1), 0), n - 1))

    bt.on_clicked(on_save)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw(n - 1)
    plt.show()
    return int(sl.val) if state["saved"] else None


# =============================================================================
# STAGES
# =============================================================================
def cfg_key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def stage_mc1(d: DayFile, path: Path, args) -> bool:
    print("\n== MC1: vessel + home ship + kappa + L ==")
    T = d.T
    sil = silences_tau(d)
    c = CONFIG
    model = MC1(T, sil, c)
    rng = np.random.default_rng(c["seed"])
    if c["seed"] is not None:
        np.random.seed(c["seed"])
    key = cfg_key("mc1", d.day, sil, {k: c[k] for k in sorted(c) if k not in ("n_show", "R_km")})

    def runner():
        Yc, lpc = model.search(rng)
        cands = pick_modes(model, Yc, lpc, c["n_refine"])
        coarse = MC1(T, sil, c, n_grid=c["n_grid_screen"])
        Yr, _ = refine_map(coarse, cands, "MC1")
        lpr = chunked(model.log_post, Yr)
        modes = pick_modes(model, np.concatenate([Yr, Yc]), np.concatenate([lpr, lpc]), c["n_modes"])
        return run_modes(model, modes, c["walkers_per_mode"], c["steps_per_mode"], rng, "MC1")

    chain, logp = cached_run(path.with_name(path.stem + ".mc1.npz"), key, runner)
    chain, logp = thin_for_slider(chain, logp, c["n_show"])
    jsb = next((b for b in d.blocks if b.kind == "JUNKY STATION" and b.start >= 0), None)

    def js_line(v, kap, L):
        if jsb is None:
            return ""
        tb, te = clock_tau(jsb.start, d.day), clock_tau(jsb.end, d.day)
        if not (0.0 <= tb < te <= 1.0):
            return f"junky station {jsb.name}: times outside DAY, MC2 cannot run\n"
        ub, thb = (float(x) for x in v.at(tb))
        ue, the = (float(x) for x in v.at(te))
        sols = [s for s in lambert_2d(ub, ue, the - thb, te - tb, kap, L, c["allow_retrograde"])
                if not (abs(s[0] - v.eta) < 1e-6 * abs(v.eta) and abs(s[1] - v.lam) < 1e-6 * abs(v.lam))]
        return (f"junky station: {len(sols)} possible orbit(s)"
                + ("" if sols else "  (MC2 impossible)") + "\n")

    def build(y, lp, i, n):
        v, h, kap, L = model.orbits(y)
        mism, xi0, xi1 = model.terms(y[None])
        runs = real_silences(v, h)
        info = (f"MC1   model {i + 1} of {n}   log posterior = {lp:.3f}\n"
                + planet_line(kap, L, d.day) + "\n\n" + orbit_table([v, h], d.day)
                + f"\nsilence mismatch = {100 * mism[0]:.2f}% of T = {mism[0] * T / 60:.1f} min\n"
                f"xi(0) = {np.sqrt(xi0[0]):.4f} R    xi(1) = {np.sqrt(xi1[0]):.4f} R\n\n"
                f"input silences: {', '.join(fmt_interval(a, b) for a, b in d.silences) or 'none'}\n"
                f"real silences:  {fmt_runs(runs, d.day)}\n"
                + js_line(v, kap, L) + "\n"
                "o launch   X vessel at return   + home ship at return\n"
                "grey band = real silence (planet between vessel and home ship)\n"
                "slider / arrow keys: choose model;  button: save")
        return dict(title=f"{d.date}  DAY {fmt_interval(*d.day)}  - MC1 candidates", day=d.day, L=L,
                    orbits=[(v, dict(lw=2.6)), (h, dict(lw=1.4))], shadows_real=runs,
                    shadows_input=sil, info=info)

    i = choose(chain, logp, build, "Orbital game - MC1", auto=args.auto)
    if i is None:
        print("  window closed without saving; MC1 will be offered again when the file changes.")
        return False
    v, h, kap, L = model.orbits(chain[i])
    mc1 = dict(day=d.day, kappa=kap, L=L, vessel=v, home=h,
               model=f"{i + 1} of {len(logp)} (sorted by log posterior); log posterior = {logp[i]:.4f}; L set to max apocentre")
    write_saved(path, mc1, None, d.saved)
    return True


def stage_mc2(d: DayFile, path: Path, args, mc1, jsb: Block) -> bool:
    print(f"\n== MC2: junky station {jsb.name} ==")
    day = mc1["day"]
    tb, te = clock_tau(jsb.start, day), clock_tau(jsb.end, day)
    if not (0.0 <= tb < te <= 1.0):
        print(f"  MC2 not run: JUNKY STATION {jsb.name} {fmt_interval(jsb.start, jsb.end)} must lie inside "
              f"the saved DAY {fmt_interval(*day)} (the vessel exists only from launch to return).")
        return False
    c = CONFIG
    model = MC2(mc1["vessel"], mc1["kappa"], mc1["L"], tb, te, c)
    rng = np.random.default_rng(c["seed"])
    key = cfg_key("mc2", day, mc1["kappa"], mc1["L"], mc1["vessel"].values(), tb, te,
                  {k: c[k] for k in sorted(c) if k not in ("n_show", "R_km")})
    def runner():
        Yc, lpc = model.search()
        modes = pick_modes(model, Yc, lpc, c["n_modes_mc2"], min_sep=1e-4)
        return run_modes(model, modes, c["walkers_per_mode_mc2"], c["steps_per_mode_mc2"], rng, "MC2")

    chain, logp = cached_run(path.with_name(path.stem + ".mc2.npz"), key, runner)
    if len(logp) == 0:
        print(f"  MC2: no station orbit inside L = {mc1['L']:.3f} R meets the vessel at both "
              f"{s_to_clock(jsb.start)} and {s_to_clock(jsb.end)}. Change the JUNKY STATION times, "
              "or delete the MC1_ lines to choose another MC1 model.")
        return False
    chain, logp = thin_for_slider(chain, logp, c["n_show"])
    v, h = mc1["vessel"], mc1["home"]
    runs = real_silences(v, h)

    def build(y, lp, i, n):
        js = model.orbit(y)
        js.name = jsb.name
        enc = js_encounters(js, v, (jsb.start, jsb.end), day)
        info = (f"MC2   model {i + 1} of {n}   log posterior = {lp:.3f}\n"
                + planet_line(mc1["kappa"], mc1["L"], day) + "\n\n" + orbit_table([v, h, js], day)
                + f"\ninput encounters: {s_to_clock(jsb.start)} and {s_to_clock(jsb.end)}\n"
                + "real encounters:  " + ", ".join(f"{tau_clock(t, day)} ({dd:.3f} R)" for t, dd in enc) + "\n\n"
                "green diamonds: encounters on the vessel path\n"
                "slider / arrow keys: choose model;  button: save")
        markers = [(tb, "jump in", "D", "tab:green"), (te, "jump off", "D", "tab:green")]
        return dict(title=f"{d.date}  junky station {jsb.name} - MC2 candidates", day=day, L=mc1["L"],
                    orbits=[(v, dict(lw=2.6)), (h, dict(lw=1.2)), (js, dict(lw=1.8))],
                    shadows_real=runs, shadows_input=silences_tau(d) if d.day == day else [],
                    blocks=[(jsb.name, tb, te, jsb.kind)], markers=markers, info=info)

    i = choose(chain, logp, build, "Orbital game - MC2", auto=args.auto)
    if i is None:
        print("  window closed without saving; MC2 will be offered again when the file changes.")
        return False
    js = model.orbit(chain[i])
    mc2 = dict(js=js, name=jsb.name, interval=(jsb.start, jsb.end),
               model=f"{i + 1} of {len(logp)} (sorted by log posterior); log posterior = {logp[i]:.4f}")
    write_saved(path, mc1, mc2, d.saved)
    return True


def build_day_scene(d: DayFile, mc1, mc2, now_tau):
    day = mc1["day"]
    v, h = mc1["vessel"], mc1["home"]
    orbits = [(v, dict(lw=2.6, now_color="white")), (h, dict(lw=1.4, now_color="0.8"))]
    markers, blocks = [], []
    js = None
    if mc2:
        js = mc2["js"]
        js.name = mc2["name"]
        orbits.append((js, dict(lw=1.8, now_color="tab:green")))
        if mc2["interval"]:
            for c, lab in zip(mc2["interval"], ("jump in", "jump off")):
                markers.append((clock_tau(c, day), lab, "D", "tab:green"))
    for b in d.blocks:
        if b.start < 0:
            continue
        ta, tb = clock_tau(b.start, day), clock_tau(b.end, day)
        blocks.append((b.name, max(ta, 0.0), min(tb, 1.0), b.kind))
        if b.kind == "CONTAINER" and 0.0 <= tb <= 1.0:
            markers.append((tb, f"container {b.name}", "s", "tab:orange"))     # MC3 placeholder
    for p in d.prophets:
        tp = clock_tau(p, day)
        if 0.0 <= tp <= 1.0:
            markers.append((tp, "", "*", "gold"))
    rem_taus = []
    for r in d.reminders:
        for o, _, _ in reminder_schedule(r, d, day):
            t = o / day_length(day)
            if 0.0 <= t <= 1.0:
                rem_taus.append(t)
    runs = real_silences(v, h)
    rows, total, avail = earnings(d)
    info = f"{d.date}   DAY {fmt_interval(*day)}   (saved)\n" + planet_line(mc1["kappa"], mc1["L"], day) + "\n"
    info += f"MC1: {mc1['model']}\n"
    if mc2:
        info += f"MC2: {mc2['model']}\n"
    info += "\n" + orbit_table([o for o, _ in orbits], day)
    info += f"\nreal silences: {fmt_runs(runs, day)}\n"
    if js is not None and mc2["interval"]:
        enc = js_encounters(js, v, mc2["interval"], day)
        info += "junky station: " + ", ".join(f"{tau_clock(t, day)}" for t, _ in enc) + "\n"
    if now_tau is not None:
        u, th = v.at(now_tau)
        ub, thb = h.at(now_tau)
        silent = bool(occluded(u, th, ub, thb))
        info += f"\nNOW {tau_clock(now_tau, day)}  tau = {now_tau:.4f}   " + ("SILENCE" if silent else "in contact") + "\n"
    info += ("\nearnings (done/available)   E emeralds  S sapphires  D diamonds\n"
             "                            B Lian balls  M Martian coins\n")
    for b, got, av in rows:
        info += f"  {b.name[:24]:<24} {fmt_earn(got, av, emoji=False)}\n"
    info += f"  {'DAY TOTAL':<24} {fmt_earn(total, avail, emoji=False)}\n"
    info += "\nmarkers: o launch  X return  D junky station  s container end (MC3)\n         * prophets  . reminders"
    return dict(title=f"Orbital game  {d.date}", day=day, L=mc1["L"], orbits=orbits,
                shadows_real=runs, shadows_input=silences_tau(d) if d.day == day else [],
                blocks=blocks, markers=markers, reminders=rem_taus, now=now_tau, info=info)


# =============================================================================
# MAIN LOOP
# =============================================================================
def print_summary(d: DayFile):
    print(f"\n--- {d.path.name if d.path else ''}  date {d.date}")
    if d.day:
        print(f"    DAY {fmt_interval(*d.day)}  T = {d.T / 3600:.3f} h")
    print(f"    SILENCE {', '.join(fmt_interval(a, b) for a, b in d.silences) or '-'}")
    print(f"    PROPHETS {', '.join(s_to_clock(p) for p in d.prophets) or '-'}")
    rows, total, avail = earnings(d)
    for b, got, av in rows:
        print(f"    {b.kind:<13} {b.name:<24} {fmt_interval(b.start, b.end) if b.start >= 0 else '[?]':<22} {fmt_earn(got, av)}")
    for r in d.reminders:
        print(f"    REMINDER {r.name} {fmt_interval(r.start, r.end)} NORMAL [{r.mu:g}, {r.sd:g}] "
              f"VOICE {r.voice}, {len(r.messages)} message(s)")
    print(f"    DAY TOTAL {fmt_earn(total, avail)}")
    for w in d.warnings:
        print(f"    WARNING: {w}")


def day_start_datetime(d: DayFile, day):
    return dt.datetime.combine(d.date, dt.time()) + dt.timedelta(seconds=day[0])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="The Orbital game v0.1 (MC1 + MC2 + day view)")
    ap.add_argument("dayfile", type=Path)
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between file checks")
    ap.add_argument("--once", action="store_true", help="one pass; show the day view and exit when closed")
    ap.add_argument("--auto", action="store_true", help="take the highest-posterior model without a slider")
    ap.add_argument("--no-voice", action="store_true", help="print reminders only")
    ap.add_argument("--snapshot", type=Path, help="save the day view to this image and exit")
    args = ap.parse_args()

    import matplotlib
    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.raise_window"] = False

    path = args.dayfile.resolve()
    last_hash = None
    fig = axes = None
    d = mc1 = mc2 = None
    schedule = []
    last_check = None

    while True:
        try:
            text, _ = read_text(path)
        except OSError as exc:
            print(f"cannot read {path}: {exc}")
            time.sleep(args.interval)
            continue
        h = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        if h != last_hash:
            last_hash = h
            d = parse_day(text, path)
            print_summary(d)
            if d.day is None:
                mc1 = None
            else:
                mc1 = saved_mc1(d.saved)
                if mc1 is None:
                    if stage_mc1(d, path, args):
                        last_hash = None
                        continue
                else:
                    jsb = next((b for b in d.blocks if b.kind == "JUNKY STATION" and b.start >= 0), None)
                    mc2 = saved_mc2(d.saved, mc1["kappa"])
                    if jsb is not None and mc2 is None:
                        if stage_mc2(d, path, args, mc1, jsb):
                            last_hash = None
                            continue
                    schedule = [s for r in d.reminders for s in reminder_schedule(r, d, mc1["day"])]
                    if fig is None or not plt.fignum_exists(fig.number):
                        if not args.snapshot:
                            plt.ion()
                        fig, axes = make_figure("Orbital game - day view")
        # ----------------------------------------------------------- per-tick updates
        if mc1 is not None and fig is not None and plt.fignum_exists(fig.number):
            day = mc1["day"]
            t0 = day_start_datetime(d, day)
            now_off = (dt.datetime.now() - t0).total_seconds()
            now_tau = now_off / day_length(day)
            render(axes, build_day_scene(d, mc1, mc2, now_tau if 0.0 <= now_tau <= 1.0 else None))
            fig.canvas.draw_idle()
            if args.snapshot:
                fig.savefig(args.snapshot, dpi=110)
                print(f"day view saved to {args.snapshot}")
                return
            if last_check is not None:
                for o, msg, r in schedule:
                    if last_check < o <= now_off:
                        speak(msg, r.voice, not args.no_voice)
            last_check = now_off
        if args.once:
            if fig is not None and plt.fignum_exists(fig.number):
                plt.ioff()
                plt.show()
            return
        if fig is not None and plt.fignum_exists(fig.number):
            fig.canvas.start_event_loop(args.interval)
        else:
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.")
