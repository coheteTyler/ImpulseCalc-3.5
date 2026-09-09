"""Acoustic and convective time scales. Never accept t=2e-6 s as a cascade."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TimeScales:
    a_m_s: float
    t_chord_convective_s: float  # c / W1
    t_chord_acoustic_s: float  # c / a
    t_domain_acoustic_s: float
    t_end_floor_s: float  # max(10*c/W1, 1.2*Lx/a)
    n_chords_at_end: float
    Mw1: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sound_speed(gamma: float, r_specific: float, t1_k: float) -> float:
    return math.sqrt(max(gamma * r_specific * t1_k, 1e-18))


def compute_times(
    *,
    chord_m: float,
    w1_m_s: float,
    gamma: float,
    r_specific: float,
    t1_k: float,
    x_in_m: float,
    x_out_m: float,
    n_chords_min: float = 10.0,
    n_lx_a_min: float = 1.2,
) -> TimeScales:
    a = sound_speed(gamma, r_specific, t1_k)
    w = max(abs(w1_m_s), 1e-9)
    c = max(chord_m, 1e-9)
    t_conv = c / w
    t_ac = c / max(a, 1e-9)
    Lx = max(x_out_m - x_in_m, c)
    t_dom = Lx / max(a, 1e-9)
    t_chords = float(n_chords_min) * t_conv
    t_ac_leave = float(n_lx_a_min) * t_dom
    t_floor = max(t_chords, t_ac_leave)
    notes = [
        f"a = sqrt(γ R T1) = sqrt({gamma} * {r_specific} * {t1_k}) = {a:.4g} m/s",
        f"t_chord_convective = c/W1 = {c}/{w} = {t_conv:.6g} s",
        f"t_chord_acoustic = c/a = {t_ac:.6g} s",
        f"t_domain_acoustic = Lx/a = {Lx:.4g}/{a:.4g} = {t_dom:.6g} s",
        f"endTime floor = max({n_chords_min}*c/W1, {n_lx_a_min}*Lx/a) = max({t_chords:.6g}, {t_ac_leave:.6g}) = {t_floor:.6g} s  (never 2e-6 s)",
        f"Mw1 = W1/a = {w/a:.4g}",
    ]
    return TimeScales(
        a_m_s=a,
        t_chord_convective_s=t_conv,
        t_chord_acoustic_s=t_ac,
        t_domain_acoustic_s=t_dom,
        t_end_floor_s=t_floor,
        n_chords_at_end=0.0,
        Mw1=w / a,
        notes=notes,
    )
