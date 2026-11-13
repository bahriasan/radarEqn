"""
Radar Detection Range Calculator
(radar equation + MOP + coherent integration + loss budget
 + antenna pattern / pointing + target RCS / geometry + polarisation)

Model
-----
Pulse energy                : E = Pt * tau
Signal bandwidth (B_sig)    : None -> 1/tau,  LFM -> chirp BW,  Phase code -> Nc/tau
Pulse compression ratio     : D = B_sig * tau
Receiver BW mismatch factor : eta = min(B_rx/B_sig, B_sig/B_rx)   (simplified)

Single-pulse SNR at range R (monostatic, same antenna for TX and RX):
    SNR1(R) = E * eta * G(theta)^2 lambda^2 sigma(aspect, pol)
              / ((4 pi)^3 k T F L_sys R^4) / L_path(R)

    G(theta)  : antenna gain toward the target, from the pattern table
                and the angle between antenna pointing and target direction
    sigma     : target RCS from the RCS table (frequency, aspect, polarisation)
    L_sys     : sum (in dB) of fixed losses (lines, radome, ...)
    L_path(R) : two-way atmospheric + rain loss
                = 2 * alpha_atm * R  +  2 * gamma_rain(pol, elev) * min(R, rain path)

Coherent integration of N pulses (N = floor(CPI / PRI)):
    SNR_N = SNR1 * N / (L_integration * L_doppler_window)

Detection range: solve SNR(R) = SNR_required for R (bisection).

Coordinates
-----------
Azimuth: degrees clockwise from north. Elevation: degrees above horizon.
Target heading: direction the target nose points / moves, clockwise from north.
Aspect angle: 0 = nose-on (target flying toward the radar), 180 = tail-on.

Tables (folder "loss_tables" next to this script, works offline)
----------------------------------------------------------------
atmos.csv         freq_ghz, db_per_km              clear-air, one-way
rain_coeffs.csv   freq_ghz, kH, alphaH, kV, alphaV ITU-R P.838-3
antenna_gain.csv  freq_ghz, angle_deg, az_cut_dbi, el_cut_dbi     (DUMMY)
rcs.csv           freq_ghz, aspect_deg, H, V, C (m^2)             (DUMMY)
radome.csv        freq_ghz, H, V, C (dB)                          (DUMMY)
Any loss key can get its own table <key>.csv with columns
    freq_ghz,value         (polarisation independent)  or
    freq_ghz,H,V,C         (per polarisation)
Tables whose comment lines contain the word DUMMY are flagged in the results.

Matched filter view (optional, needs numpy + matplotlib)
--------------------------------------------------------
Target speed, Doppler found, unambiguous range / Doppler
--------------------------------------------------------
The target flies level along its heading at the given speed. The closing
speed v*cos(elevation)*cos(aspect) gives the Doppler 2*v_r/lambda (+ =
approaching); the tangential part gives the LOS angular rate only. The
TARGET section shows what the radar finds: PRI Stable gives the apparent
(folded) range and Doppler; PRI Switcher resolves both from the different
PRIs / PRFs. Unambiguous range is c*PRI/2 (c*LCM/2 when switching) and the
unambiguous Doppler is +-PRF/2 (+-1/(2*GCD(PRI)) when switching).

CPI by time or by pulse count, Doppler
--------------------------------------
"CPI defined by" selects which one you enter: the CPI time (N = floor(CPI /
PRI) pulses) or the pulse count N (CPI = N x PRI); the other one is computed
and shown. PRF = 1 / PRI is shown as well. Doppler resolution is 1 / (N x
PRI), accuracy is the Cramer-Rao bound at the post-integration SNR, and the
FFT length is the next power of two >= N.

PRI Stable / PRI Switcher
-------------------------
PRI Stable: one PRI. PRI Switcher: a list of PRIs (us, comma separated), each
held for one CPI (the CPI input is the dwell time per PRI). Coherent
integration only happens inside a dwell, where RF and PRI are constant:
N = floor(CPI / PRI) pulses per dwell, so longer PRIs integrate fewer pulses.
Dwells are not combined; they resolve the range ambiguity: the apparent
(folded) range R mod c*PRI/2 is measured in each PRI and matched (tolerance =
range resolution), which extends the unambiguous range to c*LCM(PRIs)/2.
An echo is also lost (eclipsed) in a PRI if it overlaps a transmit pulse.
Detection is judged per dwell; the limiting dwell (lowest net integration
gain) is reported as the headline detection range.

2D geometry view (optional, needs numpy + matplotlib)
-----------------------------------------------------
Plan view (top-down, north up), side view (vertical plane through the target),
antenna gain cuts and target RCS vs aspect. Everything is drawn from the
current inputs and tables and updates live:
  - antenna lobe = gain from the antenna_gain table for the current pointing
    (plan view: azimuth sweep at the target elevation; side view: elevation
    sweep at the target azimuth), so the lobe passes through the gain that
    the radar equation actually uses toward the target
  - target RCS lobe = RCS table vs aspect, oriented with the target heading
Lobes are shown on a dB scale (antenna 40 dB, RCS 30 dB span) and are not to
scale in range. With manual gain / RCS there is no pattern, so none is drawn.

Signal chain / phase noise (inputs: "Transmitter" section, results:
"Signal chain" tab). Oscillators are given by their full L(f) (datasheet
points, osc_structure); amplifier / magnetron / COHO terms see PhaseChain.
-----------------------------------------------
Transmitter type: coherent (source oscillator -> optional SSPA / TWT /
Klystron / CFA, RX LO shared with the source or independent) or magnetron
(random phase every pulse, COHO locked to each pulse, STALO). Every block
adds an independent phase process to the slow-time samples (see
PhaseChain): shared-oscillator noise cancels at short range (range
correlation), amplifier / COHO jitter and independent LOs do not. The
magnetron pulse-to-pulse frequency jitter does NOT grow with range: the echo
carries the magnetron phase it was locked to, only the slope across the
pulse remains (pi df PW, COHO locked at the leading edge). Results: the
coherent loss of the target (in the loss budget, range dependent in the
detection range), the Doppler phase-noise floor (dBc, clutter limit) and
the Doppler line width.

Matched filter view (optional, needs numpy + matplotlib)
--------------------------------------------------------
Uses only the defined waveform: PW, MOP (None / LFM chirp bandwidth /
phase code chips) and receiver bandwidth B_rx. The received pulse plus
complex white Gaussian noise passes the B_rx front end and a matched filter
whose replica is the transmitted pulse band-limited to the same B_rx.
The signal amplitude is calibrated so that the matched filter output peak
SNR equals the single-pulse (or integrated) SNR at the target range from the
radar equation.
"""

import csv
import math
import textwrap
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

try:
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    HAVE_PLOT = True
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
except ImportError:          # the calculator works without the plot
    HAVE_PLOT = False

# Constants
C = 299_792_458.0
K = 1.380649e-23
FOUR_PI_CUBED = (4 * math.pi) ** 3

MOP_OPTIONS = ("None", "LFM", "Phase code")
MOP_PARAM_LABEL = {
    "LFM": "Chirp bandwidth (MHz)",
    "Phase code": "Number of chips",
}
MOP_PARAM_DEFAULT = {"LFM": "10", "Phase code": "13"}

PRI_MODES = ("PRI Stable", "PRI Switcher")
CPI_MODES = ("CPI time", "Pulse count")
PW_MODES = ("Pulse width", "Duty cycle")
PRI_LIST_DEFAULT = "100, 110, 121"          # us, used when PRI Switcher is chosen
STATUS_COLOR = {"ok": "#1a7f37", "warn": "#b9770e", "bad": "#c0392b"}

POL_OPTIONS = ("H", "V", "Circular")
POL_CODE = {"H": "H", "V": "V", "Circular": "C"}
POL_TILT_DEG = {"H": 0.0, "V": 90.0, "C": 45.0}   # ITU-R P.838 tau

# key, label, default (dB), typical value text, scope
#   scope: "all"   -> applies to single pulse and coherent range
#          "mop"   -> applies only when pulse compression (MOP) is used
#          "integ" -> applies only to coherent integration (N > 1)
LOSS_FIELDS = [
    ("tx_line",    "TX line loss (dB)",           "1.0", "0.5-2",        "all"),
    ("rx_line",    "RX line loss (dB)",           "1.0", "0.5-2",        "all"),
    ("radome",     "Radome loss, 2-way (dB)",     "1.0", "0.5-1.5",      "all"),
    ("beam_shape", "Beam shape loss (dB)",        "0",   "0 if pointing modeled, 1.6 scan", "all"),
    ("mf",         "Matched filter loss (dB)",    "0.8", "0.5-1",        "all"),
    ("pc_window",  "PC window loss (dB)",         "1.3", "Hamming 1.3, Taylor 1.1", "mop"),
    ("straddle",   "Straddle loss (dB)",          "1.0", "1-3 (range+Doppler)", "all"),
    ("cfar",       "CFAR loss (dB)",              "1.0", "0.5-2",        "all"),
    ("other",      "Other losses (dB)",           "0",   "-",            "all"),
    ("integration", "Integration loss (dB)",      "0.5", "0-1 (coherent)", "integ"),
    ("doppler_window", "Doppler window loss (dB)", "1.3", "Hamming 1.3", "integ"),
]
ATMOS_FIELD = ("atmos", "Atmospheric atten., 1-way (dB/km)", "0.013",
               "X-band clear 0.01-0.02")
NF_FIELD = ("nf", "Noise figure (dB)", "10", "LNA 1-2.2 (without line/radome loss)")
RAIN_FIELDS = [
    ("rain_rate", "Rain rate (mm/h)", "0", "4 moderate, 16 heavy"),
    ("rain_path", "Rain path length (km, 0 = full)", "0", "-"),
]

# Entry key -> table name (when different)
TABLE_NAME = {"gain": "antenna_gain", "rcs": "rcs"}

# Table structure by name; anything else is a "freq" table
TABLE_KIND = {
    "rain_coeffs": "rain",
    "antenna_gain": "grid",
    "rcs": "grid",
}
GRID_COLUMNS = {"antenna_gain": 4, "rcs": 5}


# -----------------------------------------------------------------
# Default tables (written to loss_tables/ on first run)
# -----------------------------------------------------------------

DUMMY_FREQS_GHZ = (2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 22, 24, 26, 28,
                   30, 32, 35, 38, 40)


def _pattern_rel_db(theta, beamwidth):
    """Dummy pattern cut: Gaussian main beam + sloping sidelobe floor."""
    main = -12.0 * (theta / beamwidth) ** 2
    floor = -25.0 - 15.0 * min(theta, 90.0) / 90.0
    return max(main, floor)


def _dummy_gain_rows():
    angles = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15,
              20, 25, 30, 40, 50, 60, 75, 90, 120, 150, 180]
    rows = []
    for f in DUMMY_FREQS_GHZ:
        g0 = 30.0 + 20.0 * math.log10(f / 10.0)     # fixed aperture
        bw_az = 4.0 * 10.0 / f                      # 3 dB beamwidths
        bw_el = 10.0 * 10.0 / f
        for a in angles:
            rows.append((f, a,
                         round(g0 + _pattern_rel_db(a, bw_az), 2),
                         round(g0 + _pattern_rel_db(a, bw_el), 2)))
    return rows


def _dummy_rcs_rows():
    aspects = [0, 15, 30, 45, 60, 75, 85, 90, 95, 105, 120, 135, 150, 165, 180]
    base_h = [1.0, 1.2, 1.8, 2.5, 3.5, 6.0, 15.0, 40.0, 15.0, 6.0,
              4.0, 3.0, 3.5, 4.5, 6.0]
    rows = []
    anchors = ((2, 0.78), (10, 1.0), (35, 1.2), (40, 1.22))   # (GHz, scale)
    for f in DUMMY_FREQS_GHZ:
        (f0, s0), (f1, s1) = next(
            (a, b) for a, b in zip(anchors, anchors[1:]) if f <= b[0])
        scale = s0 + (s1 - s0) * (math.log10(f / f0) / math.log10(f1 / f0))
        for a, h in zip(aspects, base_h):
            h = h * scale
            rows.append((f, a, round(h, 4), round(0.85 * h, 4),
                         round(0.6 * h, 4)))
    return rows


DEFAULT_TABLES = {
    "atmos": {
        "comment": [
            "Clear-air gaseous attenuation, ONE-WAY, dB/km (oxygen + water vapour)",
            "Computed with ITU-Rpy 0.4.0 (itur.models.itu676 gamma0_exact + gammaw_exact),",
            "which implements Recommendation ITU-R P.676-12 line-by-line.",
            "Reference atmosphere: 1013.25 hPa, 15 C (288.15 K), 7.5 g/m^3 water vapour.",
            "Gaseous attenuation does not depend on polarisation.",
        ],
        "header": ["freq_ghz", "db_per_km"],
        "rows": [
            (1, 0.00544),
            (2, 0.00692),
            (3, 0.00754),
            (4, 0.00809),
            (5, 0.00871),
            (6, 0.00946),
            (7, 0.01035),
            (8, 0.01142),
            (9, 0.01268),
            (10, 0.0142),
            (11, 0.01602),
            (12, 0.01823),
            (13, 0.02098),
            (14, 0.02447),
            (15, 0.02906),
            (16, 0.03532),
            (17, 0.04423),
            (18, 0.05751),
            (19, 0.07792),
            (20, 0.10893),
            (21, 0.15043),
            (22, 0.18734),
            (22.235, 0.19227),
            (23, 0.19429),
            (24, 0.17316),
            (25, 0.14623),
            (26, 0.12503),
            (27, 0.11074),
            (28, 0.10176),
            (29, 0.09649),
            (30, 0.09382),
            (31, 0.09302),
            (32, 0.09362),
            (33, 0.09533),
            (34, 0.09798),
            (35, 0.10146),
            (36, 0.10572),
            (37, 0.11076),
            (38, 0.11662),
            (39, 0.12335),
            (40, 0.13109),
            (50, 0.38843),
            (60, 14.7783),
            (80, 0.34237),
            (100, 0.45806),
        ],
    },
    "rain_coeffs": {
        "comment": [
            "Rain specific attenuation coefficients, ITU-R P.838-3",
            "gamma [dB/km] = k * R^alpha, R = rain rate (mm/h)",
            "Computed with ITU-Rpy 0.4.0 (itur.models.itu838, P.838-3 coefficients),",
            "horizontal (tau = 0) and vertical (tau = 90) polarisation, theta = 0.",
            "Polarisation / elevation combination per P.838 (tau, theta).",
        ],
        "header": ["freq_ghz", "kH", "alphaH", "kV", "alphaV"],
        "rows": [
            (1, 2.58927e-05, 0.969074, 3.07974e-05, 0.859221),
            (2, 8.46869e-05, 1.06642, 9.97661e-05, 0.948961),
            (3, 0.000138979, 1.23216, 0.000194232, 1.06876),
            (4, 0.000107135, 1.60088, 0.000246077, 1.24755),
            (5, 0.00021615, 1.69693, 0.000242764, 1.53173),
            (6, 0.000705587, 1.59005, 0.000487825, 1.57276),
            (8, 0.00411543, 1.39051, 0.00344982, 1.37974),
            (10, 0.012167, 1.2571, 0.0112919, 1.21565),
            (12, 0.0238578, 1.18247, 0.0245483, 1.12159),
            (15, 0.0448146, 1.12328, 0.0500825, 1.04399),
            (18, 0.0707841, 1.08183, 0.0770761, 1.0025),
            (20, 0.0916427, 1.05678, 0.0961112, 0.98469),
            (25, 0.15709, 0.999128, 0.153269, 0.949132),
            (30, 0.240308, 0.948457, 0.22909, 0.912923),
            (35, 0.337387, 0.904713, 0.322376, 0.876142),
            (40, 0.443057, 0.867306, 0.427375, 0.842053),
            (50, 0.659958, 0.808352, 0.647215, 0.787136),
            (60, 0.860613, 0.765632, 0.85152, 0.748565),
            (80, 1.17045, 0.711495, 1.16683, 0.702076),
            (100, 1.36711, 0.68145, 1.36805, 0.676541),
        ],
    },
    "antenna_gain": {
        "comment": [
            "DUMMY antenna pattern - replace with measured data.",
            "Principal-plane cuts vs off-boresight angle (0-180 deg).",
            "Gain toward a direction = az_cut(az_off) + el_cut(el_off) - peak",
            "Dummy: fixed aperture, 30 dBi / 4 x 10 deg beam at 10 GHz.",
        ],
        "header": ["freq_ghz", "angle_deg", "az_cut_dbi", "el_cut_dbi"],
        "rows": _dummy_gain_rows(),
    },
    "rcs": {
        "comment": [
            "DUMMY target RCS (m^2) - replace with real target data.",
            "aspect_deg: 0 = nose-on, 90 = broadside, 180 = tail-on.",
            "Columns per polarisation: H, V, C (circular).",
        ],
        "header": ["freq_ghz", "aspect_deg", "H", "V", "C"],
        "rows": _dummy_rcs_rows(),
    },
}


def default_table_dir():
    try:
        return Path(__file__).resolve().parent / "loss_tables"
    except NameError:
        return Path.cwd() / "loss_tables"


# =================================================================
# TABLES (pure, no GUI)
# =================================================================

def write_table(path, header, rows, comment=()):
    with open(path, "w", newline="", encoding="utf-8") as f:
        for line in comment:
            f.write(f"# {line}\n")
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def read_table(path):
    """Read a CSV table. Returns (header, rows, comment_lines)."""
    header, rows, comments = None, [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for raw in csv.reader(f):
            if not raw or not raw[0].strip():
                continue
            if raw[0].strip().startswith("#"):
                comments.append(",".join(raw))
                continue
            cells = [c.strip() for c in raw]
            if header is None:
                header = cells
                continue
            try:
                row = tuple(float(c.replace(",", ".")) for c in cells)
            except ValueError:
                raise ValueError(f"non-numeric row {cells}") from None
            if len(row) != len(header):
                raise ValueError(f"row {cells} does not match header")
            rows.append(row)
    if header is None or len(rows) < 2:
        raise ValueError("needs a header and at least 2 data rows")
    return header, rows, comments


def prepare_table(name, header, rows):
    """Validate and convert raw rows into a lookup structure."""
    kind = TABLE_KIND.get(name, "freq")
    if any(r[0] <= 0 for r in rows):
        raise ValueError("frequencies must be positive")

    if kind == "freq":
        rows = sorted(rows, key=lambda r: r[0])
        xs = [r[0] for r in rows]
        if any(b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError("duplicate frequency values")
        names = [h.strip().upper() for h in header[1:]]
        if len(names) == 1:
            cols = {"value": [r[1] for r in rows]}
        elif sorted(names) == ["C", "H", "V"]:
            cols = {n: [r[i + 1] for r in rows] for i, n in enumerate(names)}
        else:
            raise ValueError("use columns freq_ghz,value or freq_ghz,H,V,C")
        return {"kind": "freq", "xs": xs, "cols": cols}

    if kind == "rain":
        expected = ["freq_ghz", "kh", "alphah", "kv", "alphav"]
        if [h.strip().lower() for h in header] != expected:
            raise ValueError("expected columns freq_ghz,kH,alphaH,kV,alphaV")
        rows = sorted(rows, key=lambda r: r[0])
        xs = [r[0] for r in rows]
        if any(b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError("duplicate frequency values")
        if any(v <= 0 for r in rows for v in (r[1], r[3])):
            raise ValueError("k values must be positive")
        return {"kind": "rain", "xs": xs,
                "kH": [r[1] for r in rows], "aH": [r[2] for r in rows],
                "kV": [r[3] for r in rows], "aV": [r[4] for r in rows]}

    # grid: freq_ghz, angle, value columns...
    if len(header) != GRID_COLUMNS[name]:
        raise ValueError(f"expected {GRID_COLUMNS[name]} columns")
    groups = {}
    for r in rows:
        groups.setdefault(r[0], []).append(r)
    data = {}
    for f, grp in groups.items():
        grp.sort(key=lambda r: r[1])
        angles = [r[1] for r in grp]
        if len(angles) < 2 or any(b <= a for a, b in zip(angles, angles[1:])):
            raise ValueError(f"{f:g} GHz needs >= 2 distinct angles")
        if angles[0] < 0:
            raise ValueError("angles must be >= 0 (pattern is symmetric)")
        cols = [[r[i] for r in grp] for i in range(2, len(header))]
        data[f] = (angles, cols)
    col_names = [h.strip().upper() for h in header[2:]]
    return {"kind": "grid", "fs": sorted(data), "data": data,
            "col_names": col_names}


def load_tables(table_dir=None):
    """
    Returns (tables, messages).
    tables: {name: lookup structure + "source" + "dummy"}.
    Built-in defaults are always available; CSV files override them.
    """
    table_dir = Path(table_dir) if table_dir else default_table_dir()
    messages = []
    tables = {}
    for name, d in DEFAULT_TABLES.items():
        t = prepare_table(name, d["header"], d["rows"])
        t["source"] = "built-in"
        t["dummy"] = any("DUMMY" in c for c in d["comment"])
        tables[name] = t

    try:
        table_dir.mkdir(parents=True, exist_ok=True)
        for name, d in DEFAULT_TABLES.items():
            path = table_dir / f"{name}.csv"
            if path.exists() and name == "rain_coeffs":
                # Upgrade the old H-only file written by the previous version
                try:
                    header, _rows, _c = read_table(path)
                    if len(header) == 3:
                        path.replace(path.with_suffix(".csv.old"))
                        messages.append(
                            "rain_coeffs.csv upgraded to H/V format "
                            "(old file kept as rain_coeffs.csv.old).")
                except (OSError, ValueError):
                    pass
            if not path.exists():
                write_table(path, d["header"], d["rows"], d["comment"])
    except OSError as exc:
        messages.append(
            f"Could not create default tables ({exc}); using built-in values."
        )

    if table_dir.is_dir():
        for path in sorted(table_dir.glob("*.csv")):
            try:
                header, rows, comments = read_table(path)
                t = prepare_table(path.stem, header, rows)
            except (OSError, ValueError, KeyError) as exc:
                messages.append(f"{path.name} ignored: {exc}")
                continue
            t["source"] = path.name
            t["dummy"] = any("DUMMY" in c for c in comments)
            tables[path.stem] = t
    return tables, messages


def interp(xs, ys, x, log_x=False, log_y=False):
    """Piecewise-linear interpolation. Returns (value, clamped)."""
    clamped = x < xs[0] or x > xs[-1]
    x = min(max(x, xs[0]), xs[-1])
    i = 0
    while i < len(xs) - 2 and x > xs[i + 1]:
        i += 1
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    if log_x:
        x0, x1, x = math.log10(x0), math.log10(x1), math.log10(x)
    if log_y:
        y0, y1 = math.log10(y0), math.log10(y1)
    t = (x - x0) / (x1 - x0)
    y = y0 + t * (y1 - y0)
    return (10 ** y if log_y else y), clamped


def _range_note(tbl, what, value, lo, hi, unit):
    return (f"{tbl['source']}: {what} {value:g} {unit} is outside the table "
            f"({lo:g}-{hi:g} {unit}); end value used.")


def table_value(tables, name, freq_ghz, pol="H"):
    """Interpolated value of a frequency table. Returns (value, note)."""
    tbl = tables[name]
    cols = tbl["cols"]
    ys = cols["value"] if "value" in cols else cols[pol]
    log_y = name == "atmos" and all(y > 0 for y in ys)
    value, clamped = interp(tbl["xs"], ys, freq_ghz, log_y=log_y)
    note = None
    if clamped:
        note = _range_note(tbl, "frequency", freq_ghz,
                           tbl["xs"][0], tbl["xs"][-1], "GHz")
    return value, note


def grid_value(tbl, freq_ghz, angle_deg, col, log=False):
    """Bilinear lookup in a grid table. Returns (value, notes)."""
    notes = []
    fs = tbl["fs"]

    def along(f):
        angles, cols = tbl["data"][f]
        ys = cols[col]
        if log:
            ys = [10 * math.log10(y) for y in ys]
        v, cl = interp(angles, ys, angle_deg)
        if cl:
            notes.append(_range_note(tbl, "angle", angle_deg,
                                     angles[0], angles[-1], "deg"))
        return v

    if len(fs) == 1:
        v = along(fs[0])
    else:
        if freq_ghz < fs[0] or freq_ghz > fs[-1]:
            notes.append(_range_note(tbl, "frequency", freq_ghz,
                                     fs[0], fs[-1], "GHz"))
        f = min(max(freq_ghz, fs[0]), fs[-1])
        i = 0
        while i < len(fs) - 2 and f > fs[i + 1]:
            i += 1
        v0, v1 = along(fs[i]), along(fs[i + 1])
        t = (f - fs[i]) / (fs[i + 1] - fs[i])
        v = v0 + t * (v1 - v0)
    if log:
        v = 10 ** (v / 10)
    return v, notes


def antenna_gain(tables, freq_ghz, off_az_deg, off_el_deg):
    """
    Gain toward a direction from principal-plane cuts:
        G = az_cut(|az_off|) + el_cut(|el_off|) - G_peak     (dB)
    Returns (gain_dbi, peak_dbi, notes).
    """
    tbl = tables["antenna_gain"]
    notes = []
    g_az, n1 = grid_value(tbl, freq_ghz, abs(off_az_deg), 0)
    g_el, n2 = grid_value(tbl, freq_ghz, abs(off_el_deg), 1)
    p_az, _ = grid_value(tbl, freq_ghz, 0.0, 0)
    p_el, _ = grid_value(tbl, freq_ghz, 0.0, 1)
    peak = max(p_az, p_el)
    gain = max(g_az + g_el - peak, peak - 60.0)
    for n in n1 + n2:
        if n not in notes:
            notes.append(n)
    return gain, peak, notes


def rcs_value(tables, freq_ghz, aspect_deg, pol):
    tbl = tables["rcs"]
    if pol not in tbl["col_names"]:
        raise ValueError(f"rcs table has no column for polarisation {pol}")
    return grid_value(tbl, freq_ghz, aspect_deg,
                      tbl["col_names"].index(pol), log=True)


def rain_attenuation(tables, freq_ghz, rate_mm_h, pol="H", elev_deg=0.0):
    """
    One-way rain specific attenuation (dB/km), ITU-R P.838-3:
        k     = [kH + kV + (kH - kV) cos^2(theta) cos(2 tau)] / 2
        alpha = [kH aH + kV aV + (kH aH - kV aV) cos^2(theta) cos(2 tau)] / (2k)
    Returns (gamma, note).
    """
    if rate_mm_h <= 0:
        return 0.0, None
    tbl = tables["rain_coeffs"]
    xs = tbl["xs"]
    kh, clamped = interp(xs, tbl["kH"], freq_ghz, log_x=True, log_y=True)
    kv, _ = interp(xs, tbl["kV"], freq_ghz, log_x=True, log_y=True)
    ah, _ = interp(xs, tbl["aH"], freq_ghz, log_x=True)
    av, _ = interp(xs, tbl["aV"], freq_ghz, log_x=True)
    c = math.cos(math.radians(elev_deg)) ** 2 * \
        math.cos(math.radians(2 * POL_TILT_DEG[pol]))
    k = (kh + kv + (kh - kv) * c) / 2
    alpha = (kh * ah + kv * av + (kh * ah - kv * av) * c) / (2 * k)
    note = None
    if clamped:
        note = _range_note(tbl, "frequency", freq_ghz, xs[0], xs[-1], "GHz")
    return k * rate_mm_h ** alpha, note


# =================================================================
# GEOMETRY (pure)
# =================================================================

def wrap180(angle):
    return (angle + 180.0) % 360.0 - 180.0


def off_boresight(tgt_az, tgt_el, ant_az, ant_el):
    """
    Target direction in antenna coordinates.
    Returns (az_off, el_off, total_off) in degrees.
    """
    a, e = math.radians(tgt_az), math.radians(tgt_el)
    v = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
    pa, pe = math.radians(ant_az), math.radians(ant_el)
    # rotate by -antenna azimuth about z
    x1 = v[0] * math.cos(pa) + v[1] * math.sin(pa)
    y1 = -v[0] * math.sin(pa) + v[1] * math.cos(pa)
    z1 = v[2]
    # rotate by antenna elevation about y
    x2 = x1 * math.cos(pe) + z1 * math.sin(pe)
    z2 = -x1 * math.sin(pe) + z1 * math.cos(pe)
    y2 = y1
    az_off = math.degrees(math.atan2(y2, x2))
    el_off = math.degrees(math.atan2(z2, math.hypot(x2, y2)))
    total = math.degrees(math.acos(max(-1.0, min(1.0, x2))))
    return az_off, el_off, total


def aspect_angle(tgt_heading, tgt_az):
    """0 = nose-on (flying toward radar), 180 = tail-on. Horizontal plane."""
    los_to_radar = tgt_az + 180.0
    return abs(wrap180(tgt_heading - los_to_radar))


# =================================================================
# RADAR CALCULATIONS (pure)
# =================================================================

def db_to_lin(db):
    return 10 ** (db / 10)


def lin_to_db(x):
    return 10 * math.log10(x)


def pulse_count(pri_us, cpi_ms):
    """Number of complete PRIs inside the CPI."""
    # Small tolerance so e.g. 10 ms / 100 us gives 100, not 99 (float error)
    n = math.floor(cpi_ms * 1000.0 / pri_us + 1e-9)
    if n < 1:
        raise ValueError("CPI must contain at least one PRI.")
    return n


def signal_bandwidth_hz(pw_us, mop, mop_param):
    """
    Bandwidth of the transmitted pulse.
    Returns (bandwidth_hz, note) where note is a warning string or None.
    """
    if pw_us <= 0:
        raise ValueError("Pulse width must be positive.")
    pw_s = pw_us * 1e-6
    natural = 1.0 / pw_s

    if mop == "None":
        return natural, None

    if mop == "LFM":
        if mop_param is None or mop_param <= 0:
            raise ValueError("Chirp bandwidth must be positive.")
        chirp = mop_param * 1e6
        if chirp < natural:
            return natural, (
                "Chirp bandwidth is below 1/PW: no pulse compression, "
                "treated as unmodulated."
            )
        return chirp, None

    if mop == "Phase code":
        if (mop_param is None or mop_param < 1
                or abs(mop_param - round(mop_param)) > 1e-9):
            raise ValueError("Number of chips must be a positive integer.")
        return round(mop_param) / pw_s, None

    raise ValueError(f"Unknown MOP: {mop}")


def make_path_loss(atmos_db_km, rain_db_km, rain_path_km):
    """Returns f(R_m) -> two-way atmospheric + rain loss in dB."""
    def path_loss(r_m):
        r_km = r_m / 1000.0
        rain_km = r_km if rain_path_km <= 0 else min(r_km, rain_path_km)
        return 2.0 * atmos_db_km * r_km + 2.0 * rain_db_km * rain_km
    return path_loss


def solve_range(snr_r4, snr_req_db, path_loss):
    """
    Solve 10log10(snr_r4 / R^4) - path_loss(R) = snr_req_db for R (m).
    snr_r4 = SNR * R^4 without path loss. path_loss is non-decreasing in R.
    """
    r_free = (snr_r4 / db_to_lin(snr_req_db)) ** 0.25
    if path_loss(r_free) == 0:
        return r_free

    def margin(r):
        return (lin_to_db(snr_r4) - 40 * math.log10(r)
                - path_loss(r) - snr_req_db)

    lo, hi = r_free * 1e-6, r_free   # margin(lo) > 0, margin(hi) <= 0
    for _ in range(200):
        mid = math.sqrt(lo * hi)      # bisection in log space
        if margin(mid) > 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def pri_lcm_us(pri_list_us):
    """
    Least common multiple of the PRIs, computed on a 1 ns grid.
    Returns (lcm_us, exact). exact is False if a PRI had to be rounded.
    """
    ns = [round(p * 1000.0) for p in pri_list_us]
    if min(ns) < 1:
        raise ValueError("PRI values must be at least 0.001 us.")
    exact = all(abs(p * 1000.0 - n) < 1e-6 for p, n in zip(pri_list_us, ns))
    lcm = ns[0]
    for v in ns[1:]:
        lcm = lcm * v // math.gcd(lcm, v)
    return lcm / 1000.0, exact


def pri_gcd_us(pri_list_us):
    """
    Greatest common divisor of the PRIs on a 1 ns grid. The PRFs repeat their
    Doppler folding together every 1 / GCD(PRI) hertz.
    Returns (gcd_us, exact).
    """
    ns = [round(p * 1000.0) for p in pri_list_us]
    if min(ns) < 1:
        raise ValueError("PRI values must be at least 0.001 us.")
    exact = all(abs(p * 1000.0 - n) < 1e-6 for p, n in zip(pri_list_us, ns))
    g = ns[0]
    for v in ns[1:]:
        g = math.gcd(g, v)
    return g / 1000.0, exact


def resolve_range(apparent_m, ru_m, span_m, tol_m, max_candidates=200_000):
    """
    Ambiguity resolution from apparent (folded) values measured in several
    PRIs (used for range in m and for Doppler in Hz). A candidate is R = a0 + k*Ru0 (0 <= R < span) that
    matches the apparent range of every other PRI within tol_m.
    Returns the list of consistent candidates (m), or None if the search
    would need too many candidates.
    """
    k_count = int(math.ceil(span_m / ru_m[0] - 1e-9))
    if k_count > max_candidates:
        return None
    found = []
    for k in range(k_count):
        r = apparent_m[0] + k * ru_m[0]
        ok = True
        for a, ru in zip(apparent_m[1:], ru_m[1:]):
            d = (r - a) % ru
            if min(d, ru - d) > tol_m:
                ok = False
                break
        if ok:
            found.append(r)
    return found


def fft_size(n):
    """Smallest power of two >= n: FFT length for n coherently processed pulses."""
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def doppler_metrics(n, pri_s, wavelength, snr_tgt_db, snr_req_db):
    """
    Doppler processing of one dwell (n pulses, constant RF and PRI).

    resolution    : df = 1 / (n * PRI) = PRF / n   (rectangular weighting)
    velocity      : v = lambda * f_D / 2
    FFT length    : next power of two >= n (zero padded, interpolates only)
    accuracy      : Cramer-Rao bound of the frequency of a complex tone in
                    white noise, n samples, post-integration SNR S:
                    sigma_f = PRF/(2 pi) * sqrt(6 / (S (n^2 - 1)))
                    (~ 0.39 / (T sqrt(S)) for large n)
    unambiguous   : Doppler +-PRF/2, velocity +-lambda*PRF/4
    """
    prf = 1.0 / pri_s
    t_coh = n * pri_s
    df = 1.0 / t_coh
    nfft = fft_size(n)
    to_v = wavelength / 2.0                       # Hz -> m/s
    out = {
        "t_coh_ms": t_coh * 1e3,
        "df_res_hz": df, "dv_res": df * to_v,
        "n_fft": nfft, "fft_pad": nfft - n,
        "fft_bin_hz": prf / nfft, "fft_bin_v": prf / nfft * to_v,
        "v_unamb": wavelength * prf / 4.0,
        "sig_f_tgt": None, "sig_v_tgt": None,
        "sig_f_thr": None, "sig_v_thr": None,
    }
    if n >= 2:
        def sigma(snr_db):
            snr = db_to_lin(snr_db)
            return prf * math.sqrt(6.0 / (snr * (n * n - 1.0))) / (2 * math.pi)
        out["sig_f_tgt"] = sigma(snr_tgt_db)
        out["sig_v_tgt"] = out["sig_f_tgt"] * to_v
        out["sig_f_thr"] = sigma(snr_req_db)
        out["sig_v_thr"] = out["sig_f_thr"] * to_v
    return out


# =================================================================
# SIGNAL CHAIN: transmitter source / amplifier / RX LO phase noise
# =================================================================

CHAIN_TYPES = {"coherent": "Coherent (source + amplifier)",
               "magnetron": "Magnetron (coherent on receive)"}
# amplifier: (phase pushing deg per 1 % supply change, supply pulse-to-pulse
# jitter % rms, additive pulse-to-pulse phase jitter deg rms).
# ROUGH starting values only - replace them with the tube / module data.
AMP_TYPES = {
    "None (source direct)": (0.0, 0.0, 0.0),
    "SSPA": (2.0, 0.05, 0.2),
    "TWT": (30.0, 0.02, 0.2),
    "Klystron": (15.0, 0.02, 0.2),
    "CFA": (2.0, 0.1, 0.5),
}
RX_LO_TYPES = {"shared": "Shared with source (same oscillator)",
               "independent": "Independent oscillator"}
# Oscillator phase noise L(f) as "offset kHz : dBc/Hz" points (datasheet
# style, SSB). ROUGH typical curves - replace them with your datasheets.
# RF oscillators referred to an X / Ku band carrier, COHO at IF (tens of MHz).
OSC_PRESETS = {
    "Low-noise STALO (multiplied crystal)":
        "0.1:-70, 1:-100, 10:-120, 100:-130, 1000:-140, 10000:-150",
    "Typical PLL synthesizer":
        "0.1:-60, 1:-85, 10:-95, 100:-100, 1000:-125, 10000:-145",
    "Free-running DRO":
        "1:-60, 10:-90, 100:-115, 1000:-135, 10000:-150",
}
COHO_PRESETS = {
    "Crystal COHO": "0.1:-100, 1:-125, 10:-145, 100:-155, 1000:-160",
    "LC COHO": "0.1:-70, 1:-95, 10:-115, 100:-135, 1000:-150",
}
CUSTOM = "Custom"
CHAIN_DEFAULT = {
    "type": "coherent",
    # source (coherent) or STALO (magnetron): phase noise L(f) points
    "src_pn": OSC_PRESETS["Typical PLL synthesizer"],
    "amp": "None (source direct)",
    "amp_push": 0.0, "amp_supply": 0.0, "amp_add": 0.0,
    "rx_lo": "shared",
    "lo_pn": OSC_PRESETS["Typical PLL synthesizer"],
    # magnetron: pulse-to-pulse carrier offset (rms) and intrapulse linear
    # frequency change end - start (rms over pulses), both in kHz
    "mag_fjit_khz": 1.0, "mag_chirp_khz": 5.0,
    # coherent-on-receive reference: "analog" COHO, "digital" (TX burst
    # sampled by the ADC, digital COHO) or "none"
    "coho_mode": "analog", "coho": True,
    # analog COHO: lock point, lock residual, IF oscillator phase noise
    "lock_point": "edge", "mag_lock": 3.0,
    "coho_pn": COHO_PRESETS["Crystal COHO"],
    # digital COHO: correction type, phase measurement error, phase memory
    "dig_corr": "replica", "dig_err": 1.0, "dig_memory": True,
}
COHO_MODES = {"analog": "Analog COHO", "digital": "Digital COHO",
              "none": "No COHO"}
LOCK_POINTS = {"edge": "Leading edge", "centre": "Pulse centre"}
DIG_CORR = {"replica": "TX replica MF", "phase": "Avg. phase only"}
UNI_KINDS = ("uniform", "trip")      # kinds that can destroy coherence


class OscPN:
    """Oscillator SSB phase noise L(f) from datasheet points, interpolated
    linearly in dB vs log offset. Below the first point the first segment
    slope is continued (clamped to -40..0 dB/decade), above the last point
    the curve stays flat (white PM floor). S_phi(f) = 2 L(f) rad^2/Hz
    (one-sided, small angle)."""

    def __init__(self, text, name="oscillator"):
        self.text = text
        self.name = name
        pts = OscPN.parse(text, name)
        self.f_hz = [p[0] * 1e3 for p in pts]
        self.l_db = [p[1] for p in pts]
        lf = [math.log10(f) for f in self.f_hz]
        slope = (self.l_db[1] - self.l_db[0]) / (lf[1] - lf[0])
        self.slope0 = min(0.0, max(-40.0, slope))
        self._lf = lf

    @staticmethod
    def parse(text, name="oscillator"):
        """'0.1:-70, 1:-100, ...' -> [(kHz, dBc/Hz), ...] (validated)."""
        pts = []
        for item in str(text).replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                f, l = (float(v) for v in item.split(":"))
            except ValueError:
                raise ValueError(f"{name} phase noise: '{item}' is not "
                                 "'offset_kHz:dBc/Hz'.") from None
            if f <= 0 or not math.isfinite(f) or not math.isfinite(l):
                raise ValueError(f"{name} phase noise: offset must be > 0.")
            if l >= 0:
                raise ValueError(f"{name} phase noise: L must be < 0 dBc/Hz.")
            pts.append((f, l))
        pts.sort()
        if len(pts) < 2:
            raise ValueError(f"{name} phase noise: give at least 2 points "
                             "(offset_kHz:dBc/Hz, ...).")
        if any(pts[i][0] == pts[i + 1][0] for i in range(len(pts) - 1)):
            raise ValueError(f"{name} phase noise: duplicate offsets.")
        return pts

    def l_dbc(self, f):
        """L(f) in dBc/Hz (numpy array or float)."""
        if HAVE_PLOT:
            x = np.log10(np.maximum(np.asarray(f, dtype=float), 1e-9))
            y = np.interp(x, self._lf, self.l_db)
            low = x < self._lf[0]
            y = np.where(low, self.l_db[0] + self.slope0 * (x - self._lf[0]),
                         y)
            return y
        x = math.log10(max(f, 1e-9))
        if x <= self._lf[0]:
            return self.l_db[0] + self.slope0 * (x - self._lf[0])
        for i in range(len(self._lf) - 1):
            if x <= self._lf[i + 1]:
                a = (x - self._lf[i]) / (self._lf[i + 1] - self._lf[i])
                return self.l_db[i] + a * (self.l_db[i + 1] - self.l_db[i])
        return self.l_db[-1]

    def s_phi(self, f):
        return 2.0 * 10 ** (self.l_dbc(f) / 10.0)

    def short(self, per_line=3):
        """Compact text of the points: '1k -85  10k -95 ...'."""
        def fk(f):
            k = f / 1e3
            return (f"{k/1e3:g}M" if k >= 1000 else
                    f"{k:g}k" if k >= 1 else f"{f:g}Hz")
        items = [f"{fk(f)} {l:g}" for f, l in zip(self.f_hz, self.l_db)]
        return "\n".join("  ".join(items[i:i + per_line])
                         for i in range(0, len(items), per_line))


def mismatch_stats(s):
    """Matched filter amplitude a = sinc(x) of a pulse whose carrier is off
    by x = df PW cycles (rect pulse, nominal replica; LFM gives about the
    same at a fixed range gate through range-Doppler coupling), x Gaussian
    with rms s: returns (E[a], E[a^2])."""
    if s <= 0:
        return 1.0, 1.0
    m = q = wsum = 0.0
    k = 1601
    for i in range(k):
        x = (-8 + 16 * i / (k - 1)) * s
        w = math.exp(-0.5 * (x / s) ** 2)
        a = 1.0 if x == 0 else math.sin(math.pi * x) / (math.pi * x)
        m += w * a
        q += w * a * a
        wsum += w
    return m / wsum, q / wsum


def magnetron_jitter_from_stability(stab, f0_hz, pw_s, ripple_hz,
                                    kind="peak"):
    """Magnetron P2P and intrapulse frequency jitter (kHz rms) from a
    fractional frequency stability figure (e.g. 7e-7, measured with a
    frequency discriminator), assuming the instability is a sinusoidal
    ripple of the supply (pushing) at ripple_hz, not synchronous with the
    PRF (random ripple phase at every pulse):
      f(t) = A sin(2 pi fr t + theta),  A = stab f0 (peak) or
                                        A = sqrt(2) stab f0 (if rms)
      P2P (pulse to pulse carrier)    : A / sqrt(2) rms
      intrapulse change end - start   : 2 A sin(pi fr PW) cos(...)
                                        -> sqrt(2) A |sin(pi fr PW)| rms
    Returns (p2p_khz, intra_khz, a_hz, fr_pw)."""
    a = stab * f0_hz * (math.sqrt(2) if kind == "rms" else 1.0)
    p2p = a / math.sqrt(2)
    intra = math.sqrt(2) * a * abs(math.sin(math.pi * ripple_hz * pw_s))
    return p2p / 1e3, intra / 1e3, a, ripple_hz * pw_s


def osc_structure(curves, n, T, tau, pw, form):
    """Phase structure function D(L) = Var(phi_k+L - phi_k), L = 0..n-1, of
    the slow-time phase produced by oscillator phase noise (numpy array).

    phi_k is the matched-filter (pulse-averaged, |H|^2 = sinc^2(f PW))
    phase of pulse k:
      form 'shared'  same oscillator at TX and RX (or a COHO re-locked at
                     TX and used at the echo): theta(t) - theta(t + tau)
                     -> S_x = S_phi 4 sin^2(pi f tau)   (range correlation)
      form 'indep'   independent TX / RX oscillators: S_x = S_tx + S_rx
    D(L) = 2 int S_x(f) (1 - cos 2 pi f L T) df. Offsets below 8 PRF are
    folded exactly into the PRF band (FFT), higher offsets alias as white
    slow-time noise. Offsets below PRF / (bins per PRF) (drift slower than
    the CPI = a Doppler shift) are ignored."""
    def s_x(f):
        s = sum(c.s_phi(f) for c in curves)
        if pw > 0:
            s = s * np.sinc(f * pw) ** 2
        return s

    mb = 1 << int(math.ceil(math.log2(max(8 * n, 32 * tau / T + 1, 64))))
    prf = 1.0 / T
    df = prf / mb
    h = 8
    idx = np.arange(1, h * mb)
    f = idx * df
    g = s_x(f)
    if form == "shared":
        g = g * 4 * np.sin(np.pi * f * tau) ** 2
    sf = np.bincount(idx % mb, weights=g * df, minlength=mb)
    d = 2 * (sf.sum() - np.real(np.fft.fft(sf))[:n])
    # above 8 PRF: white in slow time
    fh = h * prf
    fmax = max(50.0 / pw if pw > 0 else 1e9, 10 * fh)
    v = 0.0
    start = fh
    if form == "shared" and tau > 0 and 10 / tau > fh:
        stop = min(10 / tau, fmax)
        step = max(1 / (16 * tau), (stop - fh) / 4000)
        f1 = np.arange(fh, stop, step)
        if len(f1) > 1:
            v += _trapz(s_x(f1) * 4 * np.sin(np.pi * f1 * tau) ** 2, f1)
        start = stop
    if fmax > start:
        f2 = np.geomspace(start, fmax, 1500)
        v += _trapz(s_x(f2) * (2.0 if form == "shared" else 1.0), f2)
    d[1:] += 2 * v
    d[0] = 0.0
    return np.maximum(d, 0.0)


def osc_structure_py(curves, n, T, tau, pw, form):
    """Pure python fallback of osc_structure (log grid, no folding)."""
    fs = [10 ** (k / 40.0) for k in range(-40, 400)]       # 0.1 Hz..10 GHz
    out = [0.0] * n
    for i in range(1, len(fs)):
        f, dfi = fs[i], fs[i] - fs[i - 1]
        s = sum(c.s_phi(f) for c in curves)
        if pw > 0:
            x = math.pi * f * pw
            s *= (math.sin(x) / x) ** 2
        if form == "shared":
            s *= 4 * math.sin(math.pi * f * tau) ** 2
        for lag in range(1, n):
            out[lag] += 2 * s * (1 - math.cos(2 * math.pi * f * lag * T)) * dfi
    return out


class PhaseChain:
    """
    Phase error of the slow-time samples (one per pulse, at the target range
    gate after down-conversion and matched filtering), built from the
    blocks of the signal chain. Every block adds an independent phase
    process; the sum is phi_k and c(L) = E[exp(j(phi_k - phi_{k+L}))] =
    prod over blocks of exp(-D_b(L)/2), D_b(L) = Var(phi_k - phi_{k+L}).

    Block kinds (T = PRI, tau = 2R/c, L = pulse lag):
      osc_shared  oscillator phase noise L(f) (datasheet points), same
                  oscillator at TX and RX (coherent source with shared LO,
                  STALO) or a COHO re-locked every pulse and used at the
                  echo: theta(t) - theta(t + tau) -> range correlation
      osc_indep   independent TX and RX oscillators: range independent
                  (both via osc_structure: full L(f), pulse averaging by
                  the matched filter, aliasing into the PRF band)
      white       independent phase every pulse, variance p:
                  D(L) = 2 p  (L > 0)             -> range independent
      uniform     random start phase each pulse (magnetron without COHO):
                  c(L) = 0 for L > 0              -> no coherent gain
      trip        magnetron + COHO, echo beyond the first trip (tau >= T):
                  the COHO has been re-locked to a later pulse with an
                  unrelated random phase -> like uniform (second-trip
                  echoes are incoherent), no effect for tau < T

    Amplifier: sigma = sqrt((pushing x supply jitter)^2 + additive^2).
    Magnetron + COHO: the COHO is phase locked to each transmitted pulse,
    leaving the lock residual. The pulse-to-pulse magnetron frequency error
    df does NOT accumulate over the round trip: the echo is the magnetron
    signal delayed by tau, so its phase at the echo centre equals the
    transmitted phase the COHO was locked to; the difference only builds up
    in the STALO / COHO over tau. What remains is the phase slope across
    the pulse: the matched filter averages the echo phase over the pulse
    (centre), the COHO is locked near the leading edge -> phase error
    pi df PW (Radar Handbook instability table), white, range independent.
    (Checked with a fast-time simulation: edge lock 1.72 deg vs pi df PW =
    1.8 deg for df = 1 kHz, PW = 10 us; centre lock ~0.) The intrapulse
    linear frequency change and the analog / digital COHO variants are in
    mag_terms(). Digital COHO: the TX burst is sampled by the receiver ADC
    every pulse (digital coherent-on-receive / burst pulse analysis); the
    echo is corrected with the measured phase or matched to the sampled TX
    pulse itself; with phase memory second-trip echoes stay coherent.
    """

    def __init__(self, spec, b_hz):
        sp = dict(CHAIN_DEFAULT)
        sp.update(spec or {})
        self.spec = sp
        self.b_hz = b_hz
        self.mag = sp["type"] == "magnetron"
        self._dcache = {}
        pw = sp.get("pw_s") or 0.0

        src = "STALO" if self.mag else "Source"
        self.src_pn = OscPN(sp["src_pn"], src)
        self.lo_pn = None
        self.coho_pn = None
        comps = []
        if self.mag or sp["rx_lo"] == "shared":
            comps.append((f"{src} phase noise (shared, range-correlated)",
                          "osc_shared", ((self.src_pn,), pw)))
        else:
            self.lo_pn = OscPN(sp["lo_pn"], "RX LO")
            comps.append(("Source phase noise (TX, independent of RX)",
                          "osc_indep", ((self.src_pn,), pw)))
            comps.append(("RX LO phase noise (independent)",
                          "osc_indep", ((self.lo_pn,), pw)))
        if self.mag:
            mode = sp.get("coho_mode") or ("analog" if sp["coho"] else "none")
            sp["coho_mode"] = mode
            sp["coho"] = mode != "none"
            if mode != "none":
                # phase error of the matched filter output (pulse average)
                # re the reference phase: p2p offset df -> slope, intrapulse
                # linear chirp dc (end - start) -> parabola (see mag_terms)
                e_f, e_c = self.mag_terms()
                comps.append(("Magnetron P2P freq. jitter", "white",
                              e_f ** 2))
                comps.append(("Magnetron intrapulse freq. jitter", "white",
                              e_c ** 2))
                # carrier offset df_k vs a nominal replica: MF amplitude
                # sinc(df PW) varies pulse to pulse (only the TX replica
                # matched filter is matched to the actual pulse)
                if not (mode == "digital" and sp["dig_corr"] == "replica"):
                    comps.append(("Magnetron P2P freq. mismatch (MF ampl.)",
                                  "amp", mismatch_stats(
                                      sp["mag_fjit_khz"] * 1e3
                                      * (sp.get("pw_s") or 0.0))))
            if mode == "analog":
                self.coho_pn = OscPN(sp["coho_pn"], "COHO")
                comps.append(("COHO lock residual", "white",
                              math.radians(sp["mag_lock"]) ** 2))
                comps.append(("COHO phase noise (lock -> echo)",
                              "osc_shared", ((self.coho_pn,), pw)))
                comps.append(("Second-trip echo (COHO re-locked)", "trip",
                              0.0))
            elif mode == "digital":
                comps.append(("Digital COHO phase measurement error",
                              "white", math.radians(sp["dig_err"]) ** 2))
                if not sp["dig_memory"]:
                    comps.append(("Second-trip echo (no phase memory)",
                                  "trip", 0.0))
            else:
                comps.append(("Magnetron random phase (no COHO)",
                              "uniform", 0.0))
        elif sp["amp"] != "None (source direct)":
            sig = math.hypot(sp["amp_push"] * sp["amp_supply"], sp["amp_add"])
            comps.append((f"{sp['amp']} pulse-to-pulse phase", "white",
                          math.radians(sig) ** 2))
        self.comps = comps
        self.ideal = all(self._pzero(k, p) for _n, k, p in comps)

    @staticmethod
    def _pzero(kind, p):
        if kind in UNI_KINDS:
            return False
        if kind in ("osc_shared", "osc_indep"):
            return False
        if kind == "amp":
            return abs(p[0] - 1) < 1e-12 and abs(p[1] - 1) < 1e-12
        return p == 0

    @staticmethod
    def _amp_factors(comps):
        """(E[a^2], E[a]^2) product of the amplitude components: the
        correlation of the pulse gains a_k at lag 0 and at lag > 0."""
        c0 = c1 = 1.0
        for _n, kind, p in comps:
            if kind == "amp":
                c0 *= p[1]
                c1 *= p[0] ** 2
        return c0, c1

    @staticmethod
    def _uni(comps, T, tau):
        """True if a component destroys the pulse-to-pulse coherence."""
        return any(k == "uniform" or (k == "trip" and tau >= T)
                   for _n, k, _p in comps)

    def _zero(self, comps, T, tau):
        """True if the components add no phase error at this range."""
        return (not self._uni(comps, T, tau)
                and all(k == "trip" or self._pzero(k, p)
                        for _n, k, p in comps))

    def oscillators(self):
        """(label, OscPN) of the oscillators in the chain."""
        out = [("STALO" if self.mag else "Source", self.src_pn)]
        if self.lo_pn is not None:
            out.append(("RX LO", self.lo_pn))
        if self.coho_pn is not None:
            out.append(("COHO (IF)", self.coho_pn))
        return out

    def mag_terms(self):
        """rms phase error (rad) of the matched filter output caused by the
        magnetron pulse-to-pulse frequency offset (sigma_f) and the
        intrapulse linear frequency change (sigma_c, end - start), for the
        selected reference. With s the time from the pulse centre, the echo
        phase re the reference is 2 pi df (s - sL) + pi dc/PW (s^2 - sL^2);
        its pulse average is -2 pi df sL + pi dc/PW (PW^2/12 - sL^2):
          lock at the leading edge (sL = -PW/2): pi sf PW, pi sc PW / 6
          lock at the centre / pulse-average phase: 0,  pi sc PW / 12
          digital TX replica matched filter:          0,  0
        (fast-time check: 1.83 / 3.03 / 1.46 deg vs 1.8 / 3.0 / 1.5)"""
        sp = self.spec
        pw = sp.get("pw_s") or 0.0
        sf = sp["mag_fjit_khz"] * 1e3
        sc = sp["mag_chirp_khz"] * 1e3
        mode = sp["coho_mode"]
        if mode == "digital":
            if sp["dig_corr"] == "replica":
                return 0.0, 0.0
            return 0.0, math.pi * sc * pw / 12
        if mode == "analog" and sp["lock_point"] == "edge":
            return math.pi * sf * pw, math.pi * sc * pw / 6
        return 0.0, math.pi * sc * pw / 12

    def amp_sigma_deg(self):
        sp = self.spec
        if self.mag or sp["amp"] == "None (source direct)":
            return 0.0
        return math.hypot(sp["amp_push"] * sp["amp_supply"], sp["amp_add"])

    def sigma_y(self, t_s):
        """Allan deviation of the source / STALO at averaging time t_s,
        from its L(f) up to the receiver bandwidth."""
        f0 = self.spec.get("f0_hz")
        if not f0 or not HAVE_PLOT:
            return None
        f = np.geomspace(1e-2, max(self.b_hz, 10.0 / t_s), 4000)
        x = np.pi * f * t_s
        sy = (f / f0) ** 2 * self.src_pn.s_phi(f)
        return math.sqrt(max(2 * _trapz(sy * np.sin(x) ** 4 / x ** 2, f),
                             0.0))

    def _d_comp(self, comp, n, T, tau):
        """D(L), L = 0..n-1, of one component (list or numpy array)."""
        _name, kind, p = comp
        if kind in ("osc_shared", "osc_indep"):
            key = (id(comp), n, T, tau)
            d = self._dcache.get(key)
            if d is None:
                curves, pw = p
                form = "shared" if kind == "osc_shared" else "indep"
                fn = osc_structure if HAVE_PLOT else osc_structure_py
                d = fn(curves, n, T, tau, pw, form)
                if len(self._dcache) > 4000:
                    self._dcache.clear()
                self._dcache[key] = d
            return d
        if kind == "white":
            return [0.0] + [2 * p] * (n - 1)
        return [0.0] * n

    def d_total(self, n, T, tau, comps=None):
        comps = self.comps if comps is None else comps
        if HAVE_PLOT:
            d = np.zeros(n)
            for c in comps:
                d += np.asarray(self._d_comp(c, n, T, tau), dtype=float)
            return d
        d = [0.0] * n
        for c in comps:
            for i, v in enumerate(self._d_comp(c, n, T, tau)):
                d[i] += v
        return d

    def dphi_rms(self, T, tau, comps=None):
        """rms phase difference between consecutive pulses (rad)."""
        comps = self.comps if comps is None else comps
        if self._uni(comps, T, tau):
            return math.pi / math.sqrt(3)      # wrapped uniform difference
        return math.sqrt(max(float(self.d_total(2, T, tau, comps)[1]), 0.0))

    def corr(self, n, T, tau, comps=None):
        """c(L), L = 0..n-1 (numpy array)."""
        comps = self.comps if comps is None else comps
        c = np.exp(-self.d_total(n, T, tau, comps) / 2)
        c0, c1 = self._amp_factors(comps)
        c[0] *= c0
        c[1:] *= c1
        if self._uni(comps, T, tau):
            c[1:] = 0.0
        return c

    def loss_db(self, n, T, tau, comps=None):
        """Coherent integration loss of n pulses (uniform weighting):
        expected power in the target filter / ideal n^2."""
        comps = self.comps if comps is None else comps
        if n < 2 or self._zero(comps, T, tau):
            return 0.0
        if HAVE_PLOT:
            c = self.corr(n, T, tau, comps)
            lags = np.arange(n)
            g = float(np.sum((n - lags) * np.where(lags == 0, 1, 2) * c))
        else:                                   # pure python fallback
            c0, c1 = self._amp_factors(comps)
            g = float(n) * c0
            if not self._uni(comps, T, tau):
                d = self.d_total(n, T, tau, comps)
                for lag in range(1, n):
                    g += 2 * (n - lag) * c1 * math.exp(-d[lag] / 2)
        return -10 * math.log10(max(g / (n * n), 1e-30))

    def phases(self, n, T, tau, rng):
        """One realisation of phi_0..phi_{n-1} (rad), relative to pulse 0:
        Gaussian with the covariance given by the structure function,
        Cov(i, j) = (D(i) + D(j) - D(|i - j|)) / 2."""
        ph = np.zeros(n)
        gauss = [c for c in self.comps if c[1] not in UNI_KINDS]
        d = self.d_total(n, T, tau, gauss)
        if n > 1 and d.max() > 0:
            i = np.arange(1, n)
            cov = (d[i][:, None] + d[i][None, :]
                   - d[np.abs(i[:, None] - i[None, :])]) / 2
            try:
                a = np.linalg.cholesky(cov + 1e-12 * np.eye(n - 1)
                                       * max(d.max(), 1e-12))
            except np.linalg.LinAlgError:
                w, v = np.linalg.eigh(cov)
                a = v * np.sqrt(np.maximum(w, 0.0))
            ph[1:] = a @ rng.standard_normal(n - 1)
        if self._uni(self.comps, T, tau):
            ph += rng.uniform(0, 2 * math.pi, n)
        return ph

    def gains(self, n, T, tau, rng):
        """One realisation of the complex pulse gains a_k exp(j phi_k):
        phase as phases(), amplitude a_k = sinc(df_k PW) of the magnetron
        frequency mismatch (independent draw)."""
        g = np.exp(1j * self.phases(n, T, tau, rng))
        sp = self.spec
        for _n, kind, _p in self.comps:
            if kind == "amp":
                s = sp["mag_fjit_khz"] * 1e3 * (sp.get("pw_s") or 0.0)
                g = g * np.sinc(s * rng.standard_normal(n))
        return g

    def spectrum(self, window, size, T, tau, shift=0.0, comps=None):
        """Expected filter-output power of a unit tone vs filter frequency
        (FFT order, `size` points over one PRF) for the weights `window`;
        the tone is at `shift` x PRF."""
        n = len(window)
        rw = np.correlate(window, window, "full")        # lags -(n-1)..n-1
        lags = np.arange(-(n - 1), n)
        cl = self.corr(n, T, tau, comps)[np.abs(lags)]
        seq = np.zeros(size, dtype=complex)
        np.add.at(seq, lags % size,
                  rw * cl * np.exp(2j * np.pi * shift * lags))
        return np.maximum(np.real(np.fft.fft(seq)), 0.0)

    def describe(self):
        """One line description of the chain."""
        sp = self.spec

        def osc(pn):
            return "L(f) " + pn.short(per_line=99) + " (offset dBc/Hz)"

        if self.mag:
            mode = sp["coho_mode"]
            if mode == "analog":
                ref = (f"analog COHO locked at the "
                       f"{LOCK_POINTS[sp['lock_point']].lower()}, residual "
                       f"{sp['mag_lock']:g} deg, COHO {osc(self.coho_pn)}")
            elif mode == "digital":
                ref = (f"digital COHO ({DIG_CORR[sp['dig_corr']]}, meas. "
                       f"error {sp['dig_err']:g} deg, phase memory "
                       + ("on" if sp["dig_memory"] else "off") + ")")
            else:
                ref = "NO COHO"
            return (f"Magnetron (P2P {sp['mag_fjit_khz']:g} kHz rms, "
                    f"intrapulse {sp['mag_chirp_khz']:g} kHz rms) -> "
                    f"antenna; {ref}; STALO {osc(self.src_pn)}")
        rx = ("RX LO = source" if sp["rx_lo"] == "shared" else
              f"RX LO independent, {osc(self.lo_pn)}")
        amp = ("" if sp["amp"] == "None (source direct)" else
               f" -> {sp['amp']} ({self.amp_sigma_deg():.3g} deg rms)")
        return f"Source {osc(self.src_pn)}{amp} -> antenna; {rx}"


def width_3db(spec):
    """-3 dB width (in bins) of the main lobe at index 0 of a periodic
    spectrum; returns len(spec) if the whole band is within 3 dB."""
    half = spec[0] / 2.0
    size = len(spec)
    below = spec < half
    if not below.any():
        return float(size)

    def edge(step):
        i = 0
        while not below[(i + step) % size]:
            i += step
        j = (i + step) % size                        # first point below
        a, b = spec[i % size], spec[j]
        return abs(i) + (a - half) / (a - b)         # linear interpolation

    return edge(1) + edge(-1)


def blackman_harris(n):
    """4-term Blackman-Harris window (-92 dB sidelobes)."""
    k = np.arange(n)
    x = 2 * np.pi * k / max(n - 1, 1)
    return (0.35875 - 0.48829 * np.cos(x) + 0.14128 * np.cos(2 * x)
            - 0.01168 * np.cos(3 * x))


def doppler_floor_dbc(chain, n, pri_s, tau_s, comps=None):
    """Phase noise floor of the Doppler spectrum relative to the target
    peak (dBc): median filter output farther than 8 bins from the line,
    computed with a -92 dB Blackman-Harris window so that the window
    sidelobes do not hide it. It is the limit on clutter cancellation /
    sub-clutter visibility set by the signal chain. None if ideal."""
    comps = chain.comps if comps is None else comps
    if n < 8 or not HAVE_PLOT or chain._zero(comps, pri_s, tau_s):
        return None
    size = 1 << max(10, (4 * fft_size(n) - 1).bit_length())
    s = chain.spectrum(blackman_harris(n), size, pri_s, tau_s, comps=comps)
    guard = int(math.ceil(8 * size / n))
    far = s[guard:size - guard]
    if len(far) == 0:
        return None
    return 10 * math.log10(max(float(np.median(far)), 1e-30) / s.max())


def jitter_metrics(n, pri_s, chain, tau_s):
    """Phase noise loss, pulse-to-pulse phase error, Doppler noise floor and
    effective Doppler resolution of one dwell."""
    out = {"jit_loss_db": 0.0, "jit_res_factor": 1.0, "jit_floor_dbc": None,
           "jit_dphi": chain.dphi_rms(pri_s, tau_s),
           "jit_comps": [(c[0], chain.loss_db(n, pri_s, tau_s, [c]),
                          chain.dphi_rms(pri_s, tau_s, [c]),
                          doppler_floor_dbc(chain, n, pri_s, tau_s, [c]))
                         for c in chain.comps]}
    if chain.ideal or n < 2:
        return out
    out["jit_loss_db"] = chain.loss_db(n, pri_s, tau_s)
    out["jit_floor_dbc"] = doppler_floor_dbc(chain, n, pri_s, tau_s)
    if HAVE_PLOT:
        size = 1 << max(12, (16 * fft_size(n) - 1).bit_length())
        size = min(size, 1 << 20)
        w = np.ones(n)
        a = width_3db(chain.spectrum(w, size, pri_s, tau_s, comps=[]))
        b = width_3db(chain.spectrum(w, size, pri_s, tau_s))
        out["jit_res_factor"] = b / a
    return out


def radar_calc(p_dbm, pri_list_us, pw_us, mop, mop_param, bw_mhz,
               freq_ghz, nf_db, snr_db, temp_k, cpi_ms,
               gain_dbi, gain_peak_dbi, rcs_m2, target_km,
               losses, atmos_db_km, rain_db_km=0.0,
               rain_rate=0.0, rain_path_km=0.0,
               sources=None, notes=None, n_pulses=None,
               speed_ms=0.0, radial_ms=0.0, chain=None):
    """
    chain         : signal chain spec (dict, keys of CHAIN_DEFAULT): source
                    / amplifier / magnetron / RX LO phase noise; see
                    PhaseChain. Gives a (possibly range dependent) coherent
                    integration loss and a wider Doppler line.
    pri_list_us   : one PRI (PRI Stable) or several PRIs (PRI Switcher).
                    Each PRI is held for one CPI (dwell). Coherent
                    integration only happens inside a dwell, where RF and
                    PRI are constant; dwells are never combined.
    cpi_ms        : CPI = dwell time per PRI: N = floor(CPI / PRI) pulses.
    n_pulses      : alternative to cpi_ms: N pulses per dwell (same N for
                    every PRI, dwell time = N x PRI). Give exactly one of
                    cpi_ms and n_pulses.
    speed_ms      : target speed (m/s), level flight along its heading
    radial_ms     : closing speed toward the radar (m/s, + = approaching) =
                    speed * cos(elevation) * cos(aspect). Only this component
                    gives Doppler; the rest is tangential (LOS angular rate).
    gain_dbi      : antenna gain toward the target (current pointing)
    gain_peak_dbi : boresight gain (antenna pointed at the target)
    losses        : dict {key: dB} for every key in LOSS_FIELDS
    sources       : optional dict {key: "manual" | "table: file.csv"}
    notes         : optional list of extra warnings

    The top-level result keys (n, int_gain_db, r_coh, margin_db, ...) are
    those of the limiting dwell (lowest net integration gain); res["dwells"]
    holds every PRI.
    """
    sources = sources or {}
    pri_list = [float(p) for p in pri_list_us]
    # ---------- validation ----------
    if not pri_list:
        raise ValueError("At least one PRI is required.")
    if min(pri_list) <= 0 or min(pw_us, bw_mhz, freq_ghz, rcs_m2, temp_k,
                                 target_km) <= 0:
        raise ValueError(
            "PRI, PW, bandwidth, frequency, RCS, temperature "
            "and target range must be positive."
        )
    if (cpi_ms is None) == (n_pulses is None):
        raise ValueError("Give either the CPI time or the pulse count.")
    if cpi_ms is not None and cpi_ms <= 0:
        raise ValueError("CPI must be positive.")
    if n_pulses is not None and (n_pulses < 1
                                 or abs(n_pulses - round(n_pulses)) > 1e-9):
        raise ValueError("Pulses per CPI must be a positive integer.")
    if speed_ms < 0:
        raise ValueError("Target speed cannot be negative.")
    if pw_us > min(pri_list):
        raise ValueError("PW cannot exceed the shortest PRI.")
    for key, label, *_ in LOSS_FIELDS:
        if losses[key] < 0:
            raise ValueError(f"{label} cannot be negative.")
    if atmos_db_km < 0 or rain_db_km < 0:
        raise ValueError("Atmospheric / rain attenuation cannot be negative.")
    if rain_rate < 0 or rain_path_km < 0:
        raise ValueError("Rain rate and rain path cannot be negative.")

    warnings = list(notes or [])
    multi = len(pri_list) > 1

    # ---------- basic quantities ----------
    peak_w = db_to_lin(p_dbm - 30)
    pw_s = pw_us * 1e-6
    wavelength = C / (freq_ghz * 1e9)
    energy_pulse = peak_w * pw_s
    f_dop = 2.0 * radial_ms / wavelength        # true Doppler (Hz), + = closing

    # ---------- waveform / MOP ----------
    b_sig, note = signal_bandwidth_hz(pw_us, mop, mop_param)
    if note:
        warnings.append(note)
    b_rx = bw_mhz * 1e6

    tbp = b_sig * pw_s                       # pulse compression ratio D
    pc_used = tbp > 1 + 1e-9
    pc_gain_db = lin_to_db(tbp)
    eta = min(b_rx / b_sig, b_sig / b_rx)    # bandwidth mismatch factor
    mismatch_db = max(0.0, -lin_to_db(eta))
    b_eff = min(b_rx, b_sig)                 # bandwidth actually processed
    compressed_width_s = 1.0 / b_eff
    range_res = C / (2 * b_eff)

    # ---------- loss budget (fixed part; integration part per dwell) ----------
    items = []           # [label, dB, note, is_integration_loss]
    fixed_db = 0.0       # applies to single pulse and coherent
    integ_total = 0.0    # applies only to dwells with N > 1
    for key, label, _default, _typ, scope in LOSS_FIELDS:
        value = losses[key]
        name = label.replace(" (dB)", "")
        src = sources.get(key, "")
        src = src if src.startswith("table") else ""
        if scope == "mop" and not pc_used:
            items.append([name, 0.0, "not applied: no pulse compression", False])
        elif scope == "integ":
            integ_total += value
            items.append([name, value,
                          "coherent only" + (f", {src}" if src else ""), True])
        else:
            fixed_db += value
            items.append([name, value, src, False])
    items.append(["BW mismatch", mismatch_db, "computed from B_rx, B_sig", False])
    fixed_db += mismatch_db

    # ---------- radar equation ----------
    def snr_r4(gain_db):
        """SNR1 * R^4 without path loss, for a given antenna gain."""
        return (energy_pulse * db_to_lin(gain_db) ** 2 * wavelength ** 2
                * rcs_m2
                / (FOUR_PI_CUBED * K * temp_k * db_to_lin(nf_db)
                   * db_to_lin(fixed_db)))

    snr1_r4 = snr_r4(gain_dbi)
    snr1_r4_bore = snr_r4(gain_peak_dbi)

    path_loss = make_path_loss(atmos_db_km, rain_db_km, rain_path_km)
    r_single = solve_range(snr1_r4, snr_db, path_loss)

    r_min = C * pw_s / 2
    r_tgt = target_km * 1000.0
    tang_ms = math.sqrt(max(speed_ms ** 2 - radial_ms ** 2, 0.0))
    los_rate_dps = math.degrees(tang_ms / r_tgt)   # LOS angular rate
    path_tgt_db = path_loss(r_tgt)
    snr1_tgt_db = lin_to_db(snr1_r4 / r_tgt ** 4) - path_tgt_db

    # ---------- one dwell per PRI ----------
    f0_hz = freq_ghz * 1e9
    tau_tgt = 2.0 * r_tgt / C                  # true round trip delay
    pchain = PhaseChain(dict(chain or {}, f0_hz=f0_hz, pw_s=pw_s), b_eff)
    dwells = []
    for p in pri_list:
        pri_s = p * 1e-6
        n = (pulse_count(p, cpi_ms) if n_pulses is None
             else int(round(n_pulses)))
        jm = jitter_metrics(n, pri_s, pchain, tau_tgt)
        integ_db = (integ_total + jm["jit_loss_db"]) if n > 1 else 0.0
        ideal_db = lin_to_db(n)
        gain_db = ideal_db - integ_db
        g_int = db_to_lin(gain_db + jm["jit_loss_db"])   # without jitter
        if not pchain.ideal and n > 1:
            # phase noise loss may depend on the round trip delay (range)
            def path_j(r_m, _p=pri_s, _n=n):
                return path_loss(r_m) + pchain.loss_db(_n, _p, 2 * r_m / C)
        else:
            path_j = path_loss
        ru = C * pri_s / 2
        apparent = r_tgt % ru                  # folded (apparent) range
        prf_i = 1.0 / pri_s
        f_app = (f_dop + prf_i / 2.0) % prf_i - prf_i / 2.0   # folded Doppler
        eclipsed = apparent < r_min or apparent > ru - r_min
        margin = snr1_tgt_db + gain_db - snr_db
        if eclipsed:
            why = "eclipsed"
        elif margin < 0:
            why = "low SNR"
        else:
            why = "OK"
        dwells.append({
            "pri_us": p, "prf_hz": 1 / pri_s, "n": n,
            "t_int_ms": n * pri_s * 1e3,
            "ideal_int_gain_db": ideal_db, "integ_db": integ_db,
            "int_gain_db": gain_db, "snr1_req_db": snr_db - gain_db,
            "duty": pw_s / pri_s, "avg_w": peak_w * pw_s / pri_s,
            "energy_cpi": energy_pulse * n,
            "r_coh": solve_range(snr1_r4 * g_int, snr_db, path_j),
            "r_coh_bore": solve_range(snr1_r4_bore * g_int, snr_db, path_j),
            **jm,
            "ru": ru, "apparent": apparent, "eclipsed": eclipsed,
            "snr_int_tgt_db": snr1_tgt_db + gain_db, "margin_db": margin,
            "detected": (not eclipsed) and margin >= 0, "why": why,
            "t_dwell_ms": cpi_ms if n_pulses is None else n * pri_s * 1e3,
            "dop_app_hz": f_app, "dop_aliased": abs(f_app - f_dop) > 1e-6,
            "range_migration_m": abs(radial_ms) * n * pri_s,
            **doppler_metrics(n, pri_s, wavelength, snr1_tgt_db + gain_db,
                              snr_db),
        })
    lim = min(range(len(dwells)), key=lambda i: dwells[i]["int_gain_db"])
    L = dwells[lim]

    # loss budget as seen by the limiting dwell
    budget = []
    for name, value, note_, is_integ in items:
        if is_integ and L["n"] == 1:
            budget.append((name, 0.0, "not applied: N = 1"))
        else:
            budget.append((name, value, note_))
    if not pchain.ideal:
        budget.append((
            "Phase noise (signal chain)",
            L["jit_loss_db"] if L["n"] > 1 else 0.0,
            f"coherent, at {r_tgt/1000:.1f} km" if L["n"] > 1
            else "not applied: N = 1"))
    for d in dwells:
        d["jit_res_factor"] = min(d["jit_res_factor"], d["n"])  # <= PRF
        d["df_eff_hz"] = d["df_res_hz"] * d["jit_res_factor"]
        d["dv_eff"] = d["dv_res"] * d["jit_res_factor"]

    # ---------- PRI switching: unambiguous range and ambiguity resolution ----------
    lcm_us, lcm_exact = pri_lcm_us(pri_list)
    ru_list = [d["ru"] for d in dwells]
    ru_total = C * lcm_us * 1e-6 / 2 if multi else ru_list[0]
    det_idx = [i for i, d in enumerate(dwells) if d["detected"]]

    resolution = {"tol_m": range_res, "candidates": None, "resolved_m": None,
                  "checked": False, "span_m": None}
    if multi and len(det_idx) >= 2:
        sub_lcm_us, _ = pri_lcm_us([pri_list[i] for i in det_idx])
        cands = resolve_range([dwells[i]["apparent"] for i in det_idx],
                              [dwells[i]["ru"] for i in det_idx],
                              C * sub_lcm_us * 1e-6 / 2, range_res)
        resolution["checked"] = cands is not None
        resolution["candidates"] = cands
        resolution["span_m"] = C * sub_lcm_us * 1e-6 / 2
        if cands is not None and len(cands) == 1:
            resolution["resolved_m"] = cands[0]

    # ---------- Doppler: unambiguous span and ambiguity resolution ----------
    gcd_us, _ = pri_gcd_us(pri_list)
    dop_span = 1e6 / gcd_us if multi else dwells[0]["prf_hz"]       # Hz
    dres = {"checked": False, "candidates": None, "resolved_hz": None,
            "tol_hz": None, "span_hz": None}
    if multi and len(det_idx) >= 2:
        sub = [dwells[i] for i in det_idx]
        g_sub, _ = pri_gcd_us([d["pri_us"] for d in sub])
        span_sub = 1e6 / g_sub
        tol_f = max(d["df_eff_hz"] for d in sub)
        cands_f = resolve_range([d["dop_app_hz"] % d["prf_hz"] for d in sub],
                                [d["prf_hz"] for d in sub], span_sub, tol_f)
        dres.update(tol_hz=tol_f, span_hz=span_sub,
                    checked=cands_f is not None, candidates=cands_f)
        if cands_f is not None and len(cands_f) == 1:
            c = cands_f[0]
            dres["resolved_hz"] = c if c < span_sub / 2 else c - span_sub

    # ---------- status ----------
    first = dwells[0]
    if not det_idx:
        reasons = sorted({d["why"] for d in dwells})
        if reasons == ["eclipsed"]:
            detail = (f"target eclipsed, apparent range "
                      f"{first['apparent']/1000:.3f} km" if not multi
                      else "target eclipsed in every PRI")
        elif reasons == ["low SNR"]:
            detail = "SNR below required"
        else:
            detail = "eclipsed or SNR below required"
        kind, status_text = "bad", f"NOT DETECTED ({detail})"
    elif not multi:
        if r_tgt <= ru_list[0]:
            kind, status_text = "ok", "DETECTED"
        else:
            kind = "warn"
            status_text = (f"DETECTED, RANGE AMBIGUOUS (apparent range "
                           f"{first['apparent']/1000:.3f} km)")
    elif len(det_idx) == 1:
        kind = "warn"
        status_text = (f"DETECTED IN 1 OF {len(dwells)} PRIs, "
                       "RANGE AMBIGUOUS")
    else:
        cands = resolution["candidates"]
        n_det = f"{len(det_idx)} OF {len(dwells)} PRIs"
        if cands is None:
            kind = "warn"
            status_text = (f"DETECTED IN {n_det}, ambiguity check skipped "
                           "(PRI set needs too many candidates)")
        elif len(cands) == 1 and r_tgt >= ru_total:
            kind = "warn"
            status_text = (f"DETECTED IN {n_det}, RANGE FOLDED (beyond the "
                           f"{ru_total/1000:,.1f} km PRI-switching range)")
        elif len(cands) == 1:
            kind = "ok"
            status_text = (f"DETECTED IN {n_det}, RANGE RESOLVED "
                           f"({cands[0]/1000:.3f} km)")
        else:
            kind = "warn"
            status_text = (f"DETECTED IN {n_det}, RANGE AMBIGUOUS "
                           f"({len(cands)} candidates)")

    # ---------- what the radar finds: range and Doppler ----------
    # unambiguous range / Doppler of the PRIs that actually detected the target
    ru_eff = resolution["span_m"] if resolution["span_m"] else ru_total
    dop_span_eff = dres["span_hz"] if dres["span_hz"] else dop_span
    first_det = dwells[det_idx[0]] if det_idx else None
    if not det_idx:
        meas = {"range_m": None, "range_state": "none",
                "dop_hz": None, "dop_state": "none"}
    elif not multi:
        meas = {"range_m": first_det["apparent"],
                "range_state": "ok" if r_tgt <= ru_list[0] else "folded",
                "dop_hz": first_det["dop_app_hz"],
                "dop_state": "aliased" if first_det["dop_aliased"] else "ok"}
    else:
        if resolution["resolved_m"] is not None:
            r_val = resolution["resolved_m"]
            r_state = "resolved" if r_tgt < ru_eff else "folded"
        else:
            r_val, r_state = first_det["apparent"], "ambiguous"
        if dres["resolved_hz"] is not None:
            d_val = dres["resolved_hz"]
            d_state = "resolved" if abs(f_dop) < dop_span_eff / 2 else "folded"
        else:
            d_val, d_state = first_det["dop_app_hz"], "ambiguous"
        meas = {"range_m": r_val, "range_state": r_state,
                "dop_hz": d_val, "dop_state": d_state}
    if kind == "ok" and meas["dop_state"] in ("aliased", "ambiguous", "folded"):
        kind = "warn"
        status_text += ", DOPPLER " + meas["dop_state"].upper()

    pointing_loss_db = 2.0 * (gain_peak_dbi - gain_dbi)   # two-way

    # ---------- warnings ----------
    if not multi:
        if r_tgt > ru_list[0]:
            warnings.append(
                f"Target is beyond the unambiguous range "
                f"({ru_list[0]/1000:.3f} km): it appears at the folded range "
                f"{first['apparent']/1000:.3f} km. Use PRI Switcher to "
                "resolve it."
            )
        if L["r_coh"] > ru_list[0]:
            warnings.append(
                "Detection range exceeds unambiguous range: targets beyond "
                "it will fold (range ambiguity)."
            )
    else:
        bad = [str(i + 1) for i, d in enumerate(dwells) if d["eclipsed"]]
        if bad:
            warnings.append(
                f"Target is eclipsed in PRI {', '.join(bad)} (echo overlaps "
                "a transmit pulse); the other PRIs are used for detection.")
        if L["r_coh"] > ru_total:
            warnings.append(
                "Detection range exceeds the PRI-switching unambiguous "
                "range (LCM of the PRIs).")
        if not lcm_exact:
            warnings.append("PRIs were rounded to 1 ns to compute the LCM.")
        if len(det_idx) >= 2 and resolution["candidates"] is not None \
                and len(resolution["candidates"]) > 1:
            warnings.append(
                "PRI set does not resolve the ambiguity within one range "
                "cell: choose PRIs that are more different from each other.")
    if pointing_loss_db > 6.0:
        warnings.append(
            f"Target is outside the antenna main beam "
            f"({pointing_loss_db:.1f} dB two-way pointing loss)."
        )
    if mismatch_db > 0.5:
        side = "wider" if b_rx > b_sig else "narrower"
        warnings.append(
            f"Receiver BW is {side} than the signal BW: "
            f"{mismatch_db:.2f} dB mismatch loss (simplified model)."
        )
    if L["n"] > 1 and L["int_gain_db"] < 0:
        warnings.append(
            "Integration-related losses exceed 10log10(N): "
            "integration reduces SNR."
        )

    if det_idx and not multi and dwells[0]["dop_aliased"]:
        d0 = dwells[0]
        warnings.append(
            f"Target Doppler {f_dop:+,.0f} Hz is outside +-PRF/2 "
            f"(+-{d0['prf_hz']/2:,.0f} Hz): it is aliased to "
            f"{d0['dop_app_hz']:+,.0f} Hz. Use PRI Switcher or a higher PRF.")
    if multi and det_idx and meas["dop_state"] == "ambiguous" \
            and len(det_idx) >= 2:
        warnings.append(
            "The PRFs do not resolve the Doppler ambiguity within one "
            "Doppler resolution cell: use PRIs that are more different.")
    mig = max(d["range_migration_m"] for d in dwells)
    if mig > range_res:
        warnings.append(
            f"Target moves {mig:.1f} m during the coherent time, more than "
            f"the range resolution ({range_res:.1f} m): range migration "
            "reduces the integration gain (not modeled).")
    if multi and abs(radial_ms) * sum(d["t_dwell_ms"] for d in dwells) \
            * 1e-3 > range_res:
        warnings.append(
            "Target moves more than a range cell during the PRI cycle: the "
            "apparent ranges of the dwells may not match (range change "
            "between dwells is not modeled).")
    one = [str(i + 1) for i, d in enumerate(dwells) if d["n"] == 1]
    if one:
        warnings.append(
            "Only 1 pulse per dwell in PRI " + ", ".join(one)
            + ": no Doppler processing (Doppler resolution = PRF).")

    return {
        "mop": mop,
        "mop_param": mop_param,
        "pw_us": pw_us,
        "snr_req_db": snr_db,
        "freq_ghz": freq_ghz,
        "peak_w": peak_w,
        "wavelength": wavelength,
        "energy_pulse": energy_pulse,
        "b_sig": b_sig,
        "b_rx": b_rx,
        "tbp": tbp,
        "pc_gain_db": pc_gain_db,
        "mismatch_db": mismatch_db,
        "compressed_width_s": compressed_width_s,
        "range_res": range_res,
        # --- PRI handling ---
        "pri_mode": "switch" if multi else "stable",
        "pri_list": pri_list,
        "cpi_mode": "time" if n_pulses is None else "count",
        "cpi_ms": cpi_ms,
        "dwells": dwells,
        "lim": lim,
        "cycle_ms": sum(d["t_dwell_ms"] for d in dwells),
        "lcm_us": lcm_us,
        "ru_total": ru_total,
        "ru_min": min(ru_list),
        "ru_max": max(ru_list),
        "resolution": resolution,
        "dop_resolution": dres,
        "meas": meas,
        "speed_ms": speed_ms,
        "radial_ms": radial_ms,
        "tang_ms": tang_ms,
        "los_rate_dps": los_rate_dps,
        "f_dop_hz": f_dop,
        "chain": pchain,
        "tau_tgt_s": tau_tgt,
        "dop_span_hz": dop_span,
        "dop_unamb_ms": wavelength * dop_span / 4.0,
        "ru_eff": ru_eff,
        "dop_span_eff_hz": dop_span_eff,
        "status_kind": kind,
        "status_text": status_text,
        # --- limiting dwell (kept for the plots and the text) ---
        "avg_w": L["avg_w"],
        "duty": L["duty"],
        "prf_hz": L["prf_hz"],
        "energy_cpi": L["energy_cpi"],
        "n": L["n"],
        "t_int_ms": L["t_int_ms"],
        "ideal_int_gain_db": L["ideal_int_gain_db"],
        "int_gain_db": L["int_gain_db"],
        "snr1_req_db": L["snr1_req_db"],
        "budget": budget,
        "fixed_db": fixed_db,
        "integ_db": L["integ_db"],
        "atmos_db_km": atmos_db_km,
        "atmos_source": sources.get("atmos", "manual"),
        "rain_db_km": rain_db_km,
        "rain_rate": rain_rate,
        "rain_path_km": rain_path_km,
        "path_single_db": path_loss(r_single),
        "path_coh_db": path_loss(L["r_coh"]),
        "gain_dbi": gain_dbi,
        "gain_peak_dbi": gain_peak_dbi,
        "gain_source": sources.get("gain", "manual"),
        "pointing_loss_db": pointing_loss_db,
        "rcs_m2": rcs_m2,
        "rcs_source": sources.get("rcs", "manual"),
        "r_single": r_single,
        "r_coh": L["r_coh"],
        "r_coh_bore": L["r_coh_bore"],
        "target_km": target_km,
        "path_tgt_db": path_tgt_db,
        "snr1_tgt_db": snr1_tgt_db,
        "snr_int_tgt_db": L["snr_int_tgt_db"],
        "margin_db": L["margin_db"],
        "detected": kind == "ok",
        "r_unamb": ru_total,
        "r_min": r_min,
        "warnings": warnings,
    }


def evaluate(inp, tables):
    """
    Resolve tables and geometry, then run the radar equation.

    inp keys:
      p_dbm, pri_mode ("stable"/"switch"), pri_list [us], pw, mop,
      mop_param, bw, freq, nf, snr, temp,
      cpi (ms, dwell per PRI) or n_pulses (N per dwell) - exactly one,
      tgt_speed (m/s, along the heading, level flight; default 0),
      pol ("H"/"V"/"C"), ant_az, ant_el,
      gain_table (bool), gain_manual,
      rcs_table (bool), rcs_manual,
      tgt_range, tgt_az, tgt_el, tgt_heading,
      loss_table {key: bool}, loss_manual {key: float or None},
      rain_rate, rain_path, notes (optional list)
    """
    notes = list(inp.get("notes", []))
    pri_list = list(inp.get("pri_list") or [inp["pri"]])
    if inp.get("pri_mode") == "switch" and len(pri_list) < 2:
        raise ValueError("PRI Switcher needs at least 2 PRIs "
                         "(comma separated list).")
    used_dummy = set()
    freq, pol = inp["freq"], inp["pol"]
    if freq <= 0:
        raise ValueError("Frequency must be positive.")
    if not -90 <= inp["tgt_el"] <= 90 or not -90 <= inp["ant_el"] <= 90:
        raise ValueError("Elevation angles must be between -90 and 90 deg.")

    def use(name):
        tbl = tables.get(name)
        if tbl is None:
            raise ValueError(f"Table {name}.csv is not available.")
        if tbl.get("dummy"):
            used_dummy.add(tbl["source"])
        return f"table: {tbl['source']}"

    sources = {}

    # ---- geometry ----
    az_off, el_off, tot_off = off_boresight(
        inp["tgt_az"], inp["tgt_el"], inp["ant_az"], inp["ant_el"])
    aspect = aspect_angle(inp["tgt_heading"], inp["tgt_az"])
    speed = float(inp.get("tgt_speed", 0.0))
    if speed < 0:
        raise ValueError("Target speed cannot be negative.")
    # closing speed: horizontal velocity along the heading projected on the
    # line of sight (3D: cos(elevation) * cos(aspect))
    radial = (speed * math.cos(math.radians(inp["tgt_el"]))
              * math.cos(math.radians(aspect)))

    # ---- antenna gain ----
    if inp["gain_table"]:
        sources["gain"] = use("antenna_gain")
        gain, peak, n_ = antenna_gain(tables, freq, az_off, el_off)
        notes += n_
    else:
        gain = peak = inp["gain_manual"]
        sources["gain"] = "manual"
        if tot_off > 0.05:
            notes.append("Manual antenna gain: antenna pointing has no "
                         "effect (constant gain).")

    # ---- RCS ----
    if inp["rcs_table"]:
        sources["rcs"] = use("rcs")
        rcs, n_ = rcs_value(tables, freq, aspect, pol)
        notes += n_
    else:
        rcs = inp["rcs_manual"]
        sources["rcs"] = "manual"

    # ---- losses ----
    values = {}
    for key in [k for k, *_ in LOSS_FIELDS] + [ATMOS_FIELD[0], NF_FIELD[0]]:
        if inp["loss_table"].get(key):
            sources[key] = use(key)
            value, note = table_value(tables, key, freq, pol)
            if note:
                notes.append(note)
        else:
            value = inp["loss_manual"][key]
            if value is None:
                raise ValueError(f"Missing value for {key}.")
            sources[key] = "manual"
        values[key] = value

    # ---- rain (polarisation and path elevation dependent) ----
    if inp["rain_rate"] < 0:
        raise ValueError("Rain rate cannot be negative.")
    if inp["rain_rate"] > 0:
        use("rain_coeffs")
    rain_db_km, note = rain_attenuation(
        tables, freq, inp["rain_rate"], pol, abs(inp["tgt_el"]))
    if note:
        notes.append(note)

    if used_dummy:
        notes.insert(0, "DUMMY tables in use: " + ", ".join(sorted(used_dummy))
                     + ". Replace them with real data.")

    res = radar_calc(
        p_dbm=inp["p_dbm"], pri_list_us=pri_list, pw_us=inp["pw"],
        mop=inp["mop"], mop_param=inp["mop_param"], bw_mhz=inp["bw"],
        freq_ghz=freq, nf_db=values["nf"], snr_db=inp["snr"],
        temp_k=inp["temp"], cpi_ms=inp.get("cpi"),
        n_pulses=inp.get("n_pulses"), speed_ms=speed, radial_ms=radial,
        chain=inp.get("chain"),
        gain_dbi=gain, gain_peak_dbi=peak, rcs_m2=rcs,
        target_km=inp["tgt_range"],
        losses={k: values[k] for k, *_ in LOSS_FIELDS},
        atmos_db_km=values["atmos"], rain_db_km=rain_db_km,
        rain_rate=inp["rain_rate"], rain_path_km=inp["rain_path"],
        sources=sources, notes=notes,
    )
    res.update({
        "pol": pol,
        "ant_az": inp["ant_az"], "ant_el": inp["ant_el"],
        "tgt_az": inp["tgt_az"], "tgt_el": inp["tgt_el"],
        "tgt_heading": inp["tgt_heading"], "aspect": aspect,
        "az_off": az_off, "el_off": el_off, "tot_off": tot_off,
        "loss_values": values,
    })
    return res


def target_summary(res, short=False):
    """Texts for the target motion and for what the radar finds."""
    lam = res["wavelength"]
    m = res["meas"]
    multi = res["pri_mode"] == "switch"
    vr = res["radial_ms"]
    sense = ("closing" if vr > 1e-9 else "receding" if vr < -1e-9
             else "no radial motion")
    tag = {"ok": "", "folded": " (folded)", "resolved": " (resolved)",
           "ambiguous": " (ambiguous)", "aliased": " (aliased)"}
    out = {
        "radial": f"{vr:+.2f} m/s ({sense})",
        "angrate": f"{res['los_rate_dps']:.3f} deg/s  "
                   f"(tangential {res['tang_ms']:.1f} m/s)",
        "dop_true": f"{res['f_dop_hz']:+,.1f} Hz",
    }
    if m["range_m"] is None:
        out["range_found"] = "not detected"
        out["dop_found"] = "not detected"
    else:
        out["range_found"] = (f"{m['range_m']/1000:,.3f} km"
                              f"{tag[m['range_state']]}")
        out["dop_found"] = (f"{m['dop_hz']:+,.1f} Hz = "
                            f"{m['dop_hz'] * lam / 2:+.2f} m/s"
                            f"{tag[m['dop_state']]}")
    if multi:
        ru, span = res["ru_eff"], res["dop_span_eff_hz"]
        partial = (abs(ru - res["r_unamb"]) > 1e-6
                   or abs(span - res["dop_span_hz"]) > 1e-6)
        what = "detected PRIs" if partial else "PRIs"
    else:
        ru, span = res["r_unamb"], res["dop_span_hz"]
    v_amb = lam * span / 4.0
    out["ru"] = (f"{ru/1000:,.1f} km" if multi else f"{ru/1000:,.3f} km")
    out["dop_unamb"] = f"+-{span/2:,.0f} Hz = +-{v_amb:.1f} m/s"
    if multi and not short:
        out["ru"] += f"  (LCM of the {what})"
        out["dop_unamb"] += f"  (1 / GCD of the {what})"
    return out


def target_measure_lines(res, line):
    """What the radar finds, for the TARGET block of the results text."""
    s = target_summary(res)
    out = [line("Range found", s["range_found"]),
           line("Doppler found", s["dop_found"])]
    if res["pri_mode"] == "switch":
        out.append(line("Apparent Doppler / PRI", "  ".join(
            f"{d['dop_app_hz']:+,.0f}" for d in res["dwells"]) + " Hz"))
        dr = res["dop_resolution"]
        if dr["candidates"] is not None and len(dr["candidates"]) > 1:
            span = dr["span_hz"]
            shown = ", ".join(
                f"{(c if c < span / 2 else c - span):+,.0f}"
                for c in dr["candidates"][:4])
            more = " ..." if len(dr["candidates"]) > 4 else ""
            out.append(line("Doppler candidates", f"{shown}{more} Hz"))
    out += [line("Unambiguous range", s["ru"]),
            line("Unambiguous Doppler", s["dop_unamb"])]
    return out


def doppler_lines(res, line):
    """DOPPLER / FFT section of the results text."""
    dw = res["dwells"]
    lim = res["lim"]
    out = ["DOPPLER / FFT"]
    if res["pri_mode"] == "stable":
        d = dw[0]
        out += [
            line("PRF", f"{d['prf_hz']:,.2f} Hz  (1 / PRI {d['pri_us']:g} us)"),
            line("Coherent time (N x PRI)",
                 f"{d['t_coh_ms']:.3f} ms  (N = {d['n']})"),
            line("Doppler resolution",
                 f"{d['df_res_hz']:,.2f} Hz = {d['dv_res']:.3f} m/s "
                 f"({d['dv_res']*3.6:.2f} km/h)"),
            line("FFT points",
                 f"{d['n_fft']}  (next power of 2 >= N, zero padding "
                 f"{d['fft_pad']})"),
            line("FFT bin spacing",
                 f"{d['fft_bin_hz']:,.2f} Hz = {d['fft_bin_v']:.3f} m/s"),
        ]
        if d["sig_f_tgt"] is not None:
            out += [
                line("Doppler accuracy @ target",
                     f"{d['sig_f_tgt']:.3g} Hz = {d['sig_v_tgt']:.3g} m/s  "
                     f"(1 sigma, SNR {d['snr_int_tgt_db']:.1f} dB)"),
                line("Doppler accuracy @ threshold",
                     f"{d['sig_f_thr']:.3g} Hz = {d['sig_v_thr']:.3g} m/s  "
                     f"(1 sigma, SNR {res['snr_req_db']:.1f} dB)"),
            ]
        else:
            out.append(line("Doppler accuracy",
                            "n/a (N = 1: no Doppler processing)"))
        out.append(line("Unambiguous Doppler",
                        f"+-{d['prf_hz']/2:,.0f} Hz = +-{d['v_unamb']:.1f} m/s "
                        f"(+-{d['v_unamb']*3.6:.0f} km/h)"))
        out += jitter_lines(res, d, line)
    else:
        out.append(
            f"  {'#':>2} {'PRF(Hz)':>10} {'Tcoh(ms)':>9} {'dF(Hz)':>8} "
            f"{'dV(m/s)':>8} {'NFFT':>6} {'sigF(Hz)':>9} {'sigV(m/s)':>10} "
            f"{'Vamb(m/s)':>10}")
        for i, d in enumerate(dw):
            sf = f"{d['sig_f_tgt']:.3g}" if d["sig_f_tgt"] is not None else "n/a"
            sv = f"{d['sig_v_tgt']:.3g}" if d["sig_v_tgt"] is not None else "n/a"
            out.append(
                f"  {i+1:>2} {d['prf_hz']:>10,.1f} {d['t_coh_ms']:>9.3f} "
                f"{d['df_res_hz']:>8.2f} {d['dv_res']:>8.3f} {d['n_fft']:>6} "
                f"{sf:>9} {sv:>10} {d['v_unamb']:>10.1f}")
        out.append("  dF = 1/Tcoh, dV = lambda*dF/2, sig = 1-sigma at the "
                   "target SNR, Vamb = +-lambda*PRF/4")
        d = dw[lim]
        if d["sig_f_thr"] is not None:
            out.append(line(
                f"Accuracy @ threshold (PRI {lim+1})",
                f"{d['sig_f_thr']:.3g} Hz = {d['sig_v_thr']:.3g} m/s  "
                f"(SNR {res['snr_req_db']:.1f} dB)"))
        if not res["chain"].ideal:
            out.append("  Phase noise per PRI:  " + ", ".join(
                f"#{i+1} {d_['jit_loss_db']:.2f} dB / x{d_['jit_res_factor']:.2f}"
                for i, d_ in enumerate(dw)) + "  (loss / resolution factor)")
        out += jitter_lines(res, d, line, f" (PRI {lim+1})")
    out.append("")
    return out


def jitter_lines(res, d, line, tag=""):
    """Signal chain phase noise effect on one dwell, for the results text."""
    ch = res["chain"]
    tau = res["tau_tgt_s"]
    pri = d["pri_us"] * 1e-6
    out = ["", "SIGNAL CHAIN / PHASE NOISE" + (f"  ({tag.strip(' ()')})"
                                           if tag else ""),
           "  " + ch.describe()]
    sy = ch.sigma_y(pri)
    if sy is not None:
        out.append(line(("STALO" if ch.mag else "Source") + " stability",
                        f"Allan dev. {sy:.3g} at tau = PRI (from L(f))"))
    out.append(line("Round trip / PRI",
                    f"{tau*1e6:,.1f} us = {tau/pri:.2f} PRI"))
    if d["n"] < 2:
        return out + [line("Phase noise effect", "n/a (N = 1)"), ""]
    def fl(v):
        return "-" if v is None else f"{v:.1f}"

    out.append(f"  {'block':<46}{'dphi(deg rms)':>14}{'loss(dB)':>10}"
               f"{'floor(dBc)':>12}")
    for name, loss, dphi, floor in d["jit_comps"]:
        out.append(f"  {name:<46}{math.degrees(dphi):>14.3g}{loss:>10.2f}"
                   f"{fl(floor):>12}")
    out += [
        f"  {'TOTAL':<46}{math.degrees(d['jit_dphi']):>14.3g}"
        f"{d['jit_loss_db']:>10.2f}{fl(d['jit_floor_dbc']):>12}",
        "  dphi  = rms phase difference of consecutive pulses (phi_k+1 - "
        "phi_k, pulse-to-pulse)",
        "          at the target range, after COHO / LO correction",
        "  loss  = coherent integration loss of the target (in the budget)",
        "  floor = Doppler phase-noise floor re the target line (dBc per "
        "Doppler filter,",
        "          median of the filters > 8 bins from the line): limits "
        "clutter",
        "          cancellation / sub-clutter visibility (strong clutter "
        "spreads to it)",
        "  oscillators: full L(f); offsets above ~1/PW are averaged by the "
        "matched filter",
        "          (they go to the range sidelobes, not to the Doppler floor)",
        line("Doppler res. w/ phase noise",
             f"{d['df_eff_hz']:,.2f} Hz = {d['dv_eff']:.3f} m/s  "
             f"(x{d['jit_res_factor']:.2f} of the ideal -3 dB width)"),
    ]
    if d["jit_res_factor"] * d["df_res_hz"] >= d["prf_hz"] * 0.99:
        out.append("  ! coherence lost: the target line fills the whole "
                   "Doppler band")
    return out


def format_result(res, table_dir=None):
    w = 28  # label column width

    def line(label, value):
        return f"  {label:<{w}}: {value}"

    def span(values, fmt):
        lo, hi = min(values), max(values)
        return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) \
            else f"{fmt.format(lo)} - {fmt.format(hi)}"

    pol_name = {"H": "H (horizontal)", "V": "V (vertical)",
                "C": "Circular"}[res["pol"]]
    switch = res["pri_mode"] == "switch"
    dw = res["dwells"]
    lim = res["lim"]
    s_t = target_summary(res)
    if res["cpi_mode"] == "time":
        dwell_txt = f"each held for one CPI = {res['cpi_ms']:g} ms"
    else:
        dwell_txt = f"each dwell = {dw[0]['n']} pulses"

    out = [
        f"TARGET STATUS: {res['status_text']}",
        f"  margin {res['margin_db']:+.2f} dB"
        + (" (worst PRI)" if switch else "")
        + f" at {res['target_km']:,.2f} km",
        "",
        "DETECTION RANGE (target direction)",
        line("Coherent, current pointing", f"{res['r_coh']/1000:,.3f} km"
             + ("  (limiting PRI)" if switch else "")),
        line("Coherent, antenna on target", f"{res['r_coh_bore']/1000:,.3f} km  (optimum)"),
    ]
    if switch:
        out.append(line("Coherent, best PRI",
                        f"{max(d['r_coh'] for d in dw)/1000:,.3f} km"))
    out += [
        line("Single pulse, current", f"{res['r_single']/1000:,.3f} km"),
        "",
    ]

    if switch:
        r = res["resolution"]
        out += [
            f"PRI SWITCHING  ({len(dw)} PRIs, {dwell_txt}, "
            f"cycle {res['cycle_ms']:g} ms)",
            f"  {'#':>2} {'PRI(us)':>9} {'N':>6} {'Gain(dB)':>9} "
            f"{'Rcoh(km)':>9} {'Ru(km)':>8} {'Apparent(km)':>13} "
            f"{'Margin(dB)':>11}  Result",
        ]
        for i, d in enumerate(dw):
            mark = " *" if i == lim else ""
            out.append(
                f"  {i+1:>2} {d['pri_us']:>9.3f} {d['n']:>6,} "
                f"{d['int_gain_db']:>9.2f} {d['r_coh']/1000:>9.2f} "
                f"{d['ru']/1000:>8.2f} {d['apparent']/1000:>13.3f} "
                f"{d['margin_db']:>11.2f}  {d['why']}{mark}")
        out += [
            "  * limiting dwell (lowest net integration gain)",
            line("PRI LCM", f"{res['lcm_us']/1000:,.3f} ms"),
            line("Unambiguous range (LCM)", f"{res['ru_total']/1000:,.2f} km"),
            line("Match tolerance", f"{r['tol_m']:,.1f} m  (range resolution)"),
        ]
        if r["resolved_m"] is not None:
            out.append(line("Resolved target range",
                            f"{r['resolved_m']/1000:,.3f} km"))
        elif r["candidates"] is not None and len(r["candidates"]) > 1:
            shown = ", ".join(f"{c/1000:,.3f}" for c in r["candidates"][:4])
            more = " ..." if len(r["candidates"]) > 4 else ""
            out.append(line("Candidate ranges", f"{shown}{more} km"))
        else:
            out.append(line("Resolved target range",
                            "not available (needs detection in >= 2 PRIs)"
                            if r["candidates"] is None and r["checked"] is False
                            and sum(d["detected"] for d in dw) < 2
                            else "not available"))
        out.append("")

    out += [
        "TARGET",
        line("Range", f"{res['target_km']:,.3f} km"),
        line("Azimuth / elevation", f"{res['tgt_az']:.2f} / {res['tgt_el']:.2f} deg"),
        line("Heading / speed", f"{res['tgt_heading']:.2f} deg / "
             f"{res['speed_ms']:g} m/s ({res['speed_ms']*3.6:.0f} km/h)"),
        line("Aspect angle", f"{res['aspect']:.2f} deg  (0 nose-on, 180 tail-on)"),
        line("Radial velocity", s_t["radial"]),
        line("LOS angular rate", s_t["angrate"]),
        line("Doppler (true)", s_t["dop_true"]),
        line("RCS", f"{res['rcs_m2']:.4g} m^2 = "
                    f"{lin_to_db(res['rcs_m2']):.2f} dBsm  ({res['rcs_source']})"),
        line("Path loss to target", f"{res['path_tgt_db']:.2f} dB  (2-way)"),
        line("Single-pulse SNR", f"{res['snr1_tgt_db']:.2f} dB"),
        line("Integrated SNR", f"{res['snr_int_tgt_db']:.2f} dB"
             + ("  (limiting PRI)" if switch else "")),
        line("Margin vs required SNR", f"{res['margin_db']:+.2f} dB"),
        *target_measure_lines(res, line),
        "",
        "ANTENNA",
        line("Pointing az / el", f"{res['ant_az']:.2f} / {res['ant_el']:.2f} deg"),
        line("Polarisation", pol_name),
        line("Off-boresight az / el",
             f"{res['az_off']:+.2f} / {res['el_off']:+.2f} deg"),
        line("Off-boresight total", f"{res['tot_off']:.2f} deg"),
        line("Gain toward target", f"{res['gain_dbi']:.2f} dBi  ({res['gain_source']})"),
        line("Peak (boresight) gain", f"{res['gain_peak_dbi']:.2f} dBi"),
        line("Pointing loss", f"{res['pointing_loss_db']:.2f} dB  (2-way)"),
        "",
        "WAVEFORM",
        line("MOP", res["mop"]),
        line("Signal bandwidth", f"{res['b_sig']/1e6:,.3f} MHz"),
        line("Receiver bandwidth", f"{res['b_rx']/1e6:,.3f} MHz"),
        line("Compression ratio (B*tau)", f"{res['tbp']:,.2f}"),
        line("Pulse compression gain", f"{res['pc_gain_db']:.2f} dB"),
        line("Compressed pulse width",
             f"{res['compressed_width_s']*1e6:,.4f} us"),
        line("Range resolution", f"{res['range_res']:,.2f} m"),
        "",
        "INTEGRATION" + (f"  (limiting dwell: PRI {lim+1} = "
                         f"{dw[lim]['pri_us']:g} us)" if switch else ""),
    ]
    out.append(line("CPI defined by",
                    "time: N = floor(CPI / PRI)" if res["cpi_mode"] == "time"
                    else "pulse count: CPI = N x PRI"))
    if switch and res["cpi_mode"] == "time":
        out.append(line("Dwell time per PRI (CPI)", f"{res['cpi_ms']:g} ms"))
    out += [
        line("Pulse count N", f"{res['n']:,}"),
        line("CPI used (N x PRI)", f"{res['t_int_ms']:.3f} ms"),
        line("Ideal gain (10log10 N)", f"{res['ideal_int_gain_db']:.2f} dB"),
        line("Integration-related losses", f"{res['integ_db']:.2f} dB"),
        line("Net integration gain", f"{res['int_gain_db']:.2f} dB"),
        line("Required single-pulse SNR", f"{res['snr1_req_db']:.2f} dB"),
        "",
    ]
    out += doppler_lines(res, line)
    out += [
        f"LOSS BUDGET  (at {res['freq_ghz']:g} GHz, {pol_name})",
    ]

    for name, value, note in res["budget"]:
        suffix = f"  ({note})" if note else ""
        out.append(line(name, f"{value:5.2f} dB{suffix}"))
    out.append("  " + "-" * 44)
    out.append(line("TOTAL LOSSES (budget)",
                    f"{sum(v for _n, v, _s in res['budget']):5.2f} dB"))
    out.append("")

    if res["rain_rate"] > 0:
        rain_txt = f"{res['rain_rate']:g} mm/h, "
        rain_txt += ("full path" if res["rain_path_km"] <= 0
                     else f"first {res['rain_path_km']:g} km")
        rain_txt += f", pol {res['pol']}"
    else:
        rain_txt = "no rain"

    out += [
        line("Atmospheric, 1-way",
             f"{res['atmos_db_km']:.4f} dB/km  ({res['atmos_source']})"),
        line("Rain, 1-way",
             f"{res['rain_db_km']:.4f} dB/km  ({rain_txt})"),
        line("Path loss @ single range", f"{res['path_single_db']:5.2f} dB  (2-way)"),
        line("Path loss @ coherent range", f"{res['path_coh_db']:5.2f} dB  (2-way)"),
        "  " + "-" * 44,
        line("Total, single pulse",
             f"{res['fixed_db'] + res['path_single_db']:5.2f} dB"),
        line("Total, coherent",
             f"{res['fixed_db'] + res['integ_db'] + res['path_coh_db']:5.2f} dB"),
        "  (pointing loss is in the antenna gain, not in this total)",
        "",
        "POWER / ENERGY",
        line("Peak power", f"{res['peak_w']:,.3f} W"),
        line("Duty cycle", span([100 * d["duty"] for d in dw], "{:.3f} %")),
        line("Average power", span([d["avg_w"] for d in dw], "{:,.3f} W")),
        line("Energy per pulse", f"{res['energy_pulse']:.4g} J"),
        line("Energy per CPI", span([d["energy_cpi"] for d in dw], "{:.4g} J")),
        line("PRF", span([d["prf_hz"] for d in dw], "{:,.2f} Hz")),
        line("Wavelength", f"{res['wavelength']:.6f} m"),
        "",
        "RANGE LIMITS",
    ]
    if switch:
        out += [
            line("Unambiguous range per PRI",
                 f"{res['ru_min']/1000:,.3f} - {res['ru_max']/1000:,.3f} km"),
            line("Unambiguous range (LCM)", f"{res['ru_total']/1000:,.2f} km"),
        ]
    else:
        out.append(line("Unambiguous range", f"{res['r_unamb']/1000:,.3f} km"))
        if res["target_km"] * 1000 > res["r_unamb"]:
            out.append(line("Apparent (folded) range",
                            f"{dw[0]['apparent']/1000:,.3f} km"))
    out.append(line("Minimum (eclipse) range", f"{res['r_min']/1000:,.3f} km"))

    if res["warnings"]:
        out += ["", "WARNINGS"] + [f"  ! {msg}" for msg in res["warnings"]]

    out += [
        "",
        "ASSUMPTIONS",
        "  - Noise-limited, free-space propagation: no multipath /",
        "    propagation factor, no clutter, no jamming.",
        "  - Monostatic: same antenna, gain and polarisation on TX and RX.",
        "  - Required SNR is entered for the chosen Pd/Pfa; target",
        "    fluctuation (Swerling) loss is not modeled unless it",
        "    is added to Other losses.",
        "  - Antenna gain toward a direction = az cut + el cut - peak (dB).",
        "  - Aspect angle in the horizontal plane only.",
        "  - Polarisation acts through the RCS table, rain attenuation",
        "    (ITU-R P.838) and any loss table with H/V/C columns;",
        "    gaseous attenuation is polarisation independent.",
        "  - BW mismatch: simplified min(B_rx/B_sig, B_sig/B_rx),",
        "    added in addition to the matched filter loss.",
        "  - Coherent integration only while RF and PRI are constant",
        "    (one CPI / dwell); ideal apart from listed losses; target",
        "    stays in one range cell during the dwell.",
        "  - PRI Switcher: each PRI is held for one CPI; dwells are not",
        "    combined, they only resolve the range ambiguity.",
        "  - Target: level flight at constant speed along its heading.",
        "    Doppler uses the radial component v*cos(el)*cos(aspect);",
        "    tangential motion gives no Doppler (monostatic) and only",
        "    sets the LOS angular rate. No clutter: blind speeds and the",
        "    zero-Doppler notch are not penalised.",
        "  - Doppler ambiguity: folded Doppler of each PRF matched within",
        "    one Doppler-resolution cell; unambiguous span 1/GCD(PRI).",
        "  - Doppler resolution = 1 / (N x PRI), rectangular weighting;",
    "    a window widens it (about 1.3x for Hamming). FFT length = next",
    "    power of two >= N (zero padding only interpolates).",
    "  - Doppler accuracy = Cramer-Rao bound for one tone in white",
    "    noise at the post-integration SNR (checked by simulation down",
    "    to 13 dB); real estimators are a bit worse, and near the",
    "    threshold rare outliers (wrong FFT bin) raise the rms error.",
    "  - Eclipsing: an echo is lost if its apparent range is within",
        "    c*PW/2 of a transmit pulse (start or end of the PRI).",
        "  - Ambiguity resolution: apparent ranges matched within one",
        "    range-resolution cell, needs detection in >= 2 PRIs.",
        "  - Atmospheric / rain loss: horizontal sea-level path with",
        "    uniform attenuation (pessimistic for elevated paths).",
    ]
    if table_dir:
        out += ["", f"Tables folder: {table_dir}"]
    return "\n".join(out)


# =================================================================
# MATCHED FILTER SIMULATION (numpy)
# =================================================================

BARKER = {
    2: [1, -1], 3: [1, 1, -1], 4: [1, 1, -1, 1], 5: [1, 1, 1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


def tx_pulse(pw_s, mop, mop_param, b_sig, fs):
    """Complex baseband transmitted pulse. Returns (t, samples, description)."""
    n = max(2, int(round(pw_s * fs)))
    t = (np.arange(n) + 0.5) / fs
    if mop == "LFM" and b_sig * pw_s > 1 + 1e-9:
        k = b_sig / pw_s                                   # chirp rate
        s = np.exp(1j * np.pi * k * (t - pw_s / 2) ** 2)
        desc = f"LFM chirp, {b_sig/1e6:g} MHz"
    elif mop == "Phase code" and mop_param and round(mop_param) > 1:
        nc = int(round(mop_param))
        code = BARKER.get(nc)
        if code is None:
            raise ValueError(
                f"The {nc}-chip code sequence is not defined. Only the chip "
                "count is an input, so the plot can only use Barker codes "
                f"(N = {', '.join(str(k) for k in sorted(BARKER))}).")
        desc = f"Barker-{nc} phase code"
        chip = np.minimum((t / pw_s * nc).astype(int), nc - 1)
        s = np.asarray(code, dtype=float)[chip].astype(complex)
    else:
        s = np.ones(n, dtype=complex)
        desc = "Rectangular pulse (no MOP)"
    return t, s, desc


def simulate_mf(res, integrated=False, seed=1, max_samples=2 ** 17):
    """
    Simulate receiver input and matched filter output around the target.

    Chain: pulse (delayed to target range) + complex white noise
           -> front-end brick-wall filter |f| <= B_rx/2
           -> matched filter: correlation with the transmitted pulse
              band-limited to the same B_rx (MF bandwidth = receiver BW)
    The amplitude is calibrated so that the noise-free matched filter
    peak / mean output noise power = SNR from the radar equation.
    Integrated mode: the coherent sum of N pulses with independent
    Gaussian noise is statistically one realisation at SNR x N.
    """
    pw_s = res["pw_us"] * 1e-6
    b_sig, b_rx = res["b_sig"], res["b_rx"]
    b_max, b_eff = max(b_sig, b_rx), min(b_sig, b_rx)
    notes = []

    fs = max(8 * b_max, 64 / pw_s)              # oversampled
    t_win = 4 * pw_s + 40 / b_eff                # observation window
    if t_win * fs > max_samples:
        fs = max(2.5 * b_max, max_samples / t_win)
    L = int(round(t_win * fs))
    if L > max_samples:
        L = max_samples
        notes.append("window shortened to limit samples")

    t_p, pulse, desc = tx_pulse(pw_s, res["mop"], res["mop_param"], b_sig, fs)
    n_p = len(pulse)
    L = max(L, 2 * n_p)

    freqs = np.fft.fftfreq(L, 1 / fs)
    h_rx = (np.abs(freqs) <= b_rx / 2).astype(float)   # receiver BW
    rep_pad = np.zeros(L, dtype=complex)
    rep_pad[:n_p] = pulse
    h_mf = np.conj(np.fft.fft(rep_pad) * h_rx)          # MF, same BW
    h_tot = h_rx * h_mf                                 # front end x MF

    i0 = L // 2 - n_p // 2                       # pulse start index
    sig = np.zeros(L, dtype=complex)
    sig[i0:i0 + n_p] = pulse
    S = np.fft.fft(sig)
    s_in = np.fft.ifft(S * h_rx)                 # signal at MF input
    y_s = np.fft.ifft(S * h_tot)                 # signal at MF output

    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)
    N = np.fft.fft(noise)
    n_in = np.fft.ifft(N * h_rx)
    y_n = np.fft.ifft(N * h_tot)

    pn_in = np.mean(h_rx ** 2)                   # expected noise powers
    pn_out = np.mean(np.abs(h_tot) ** 2)

    snr_db = res["snr_int_tgt_db"] if integrated else res["snr1_tgt_db"]
    thr_db = res["snr_req_db"] if integrated else res["snr1_req_db"]
    ps = np.max(np.abs(y_s) ** 2)
    amp = np.sqrt(10 ** (snr_db / 10) * pn_out / ps)

    def db(x):
        return 10 * np.log10(np.maximum(x, 1e-12))

    in_clean = db(np.abs(amp * s_in) ** 2 / pn_in)
    in_total = db(np.abs(amp * s_in + n_in) ** 2 / pn_in)
    out_clean = db(np.abs(amp * y_s) ** 2 / pn_out)
    out_total = db(np.abs(amp * y_s + y_n) ** 2 / pn_out)

    delay = 2 * res["target_km"] * 1000.0 / C
    t0 = delay - i0 / fs
    range_km = C * (t0 + np.arange(L) / fs) / 2 / 1000.0

    # measured output peak near the target
    half = max(3, int(round(2 * fs / b_eff)))
    lo, hi = max(0, i0 - half), min(L, i0 + half)
    measured_peak = float(np.max(out_total[lo:hi]))

    zoom_km = min(25 * C / (2 * b_eff), C * pw_s / 2 * 2) / 1000.0
    zoom_km = min(zoom_km, (range_km[-1] - range_km[0]) / 2)

    return {
        "desc": desc, "fs": fs, "notes": notes,
        "zoom_km": zoom_km,
        "t_pulse_us": t_p * 1e6, "pulse": pulse,
        "range_km": range_km,
        "in_clean": in_clean, "in_total": in_total,
        "out_clean": out_clean, "out_total": out_total,
        "snr_db": snr_db, "thr_db": thr_db,
        "input_peak_snr_db": float(np.max(in_clean)),
        "measured_peak_db": measured_peak,
        "integrated": integrated,
    }


def draw_mf(fig, res, sim):
    """Draw pulse, MF input and MF output into a matplotlib figure."""
    fig.clear()
    ax1, ax2, ax3 = fig.subplots(3, 1, gridspec_kw={"height_ratios": [1, 1.3, 1.6]})
    c_noise, c_sig = "#4a90c2", "#e67e22"

    # 1) transmitted pulse
    ax1.plot(sim["t_pulse_us"], sim["pulse"].real, color="#4a90c2",
             lw=0.8, label="I (real part)")
    ax1.plot(sim["t_pulse_us"], np.abs(sim["pulse"]), color="black",
             lw=1.2, label="envelope")
    ax1.set_title(f"Transmitted pulse: {sim['desc']}, "
                  f"PW = {res['pw_us']:g} us", fontsize=9)
    ax1.set_xlabel("time (us)", fontsize=8)
    ax1.set_ylim(-1.3, 1.3)
    ax1.legend(loc="upper right", fontsize=7)

    x = sim["range_km"]
    floor_kw = dict(color="gray", ls="--", lw=1.2)

    # 2) matched filter input
    ax2.plot(x, sim["in_total"], color=c_noise, lw=0.6, alpha=0.8,
             label="signal + noise")
    ax2.plot(x, sim["in_clean"], color=c_sig, lw=1.5, label="signal only",
             zorder=3)
    ax2.axhline(0, label="noise floor (mean power)", **floor_kw)
    ax2.set_title(f"Matched filter INPUT (after {res['b_rx']/1e6:g} MHz "
                  f"front end)\npeak SNR {sim['input_peak_snr_db']:.1f} dB",
                  fontsize=9)

    # 3) matched filter output
    ax3.plot(x, sim["out_total"], color=c_noise, lw=0.6, alpha=0.8,
             label="signal + noise")
    ax3.plot(x, sim["out_clean"], color=c_sig, lw=1.5, label="signal only",
             zorder=3)
    ax3.axhline(0, label="noise floor (mean power)", **floor_kw)
    ax3.axhline(sim["thr_db"], color="#c0392b", ls="--", lw=1.2,
                label=f"required SNR ({sim['thr_db']:.1f} dB)")
    lim_note = (f", limiting PRI {res['pri_list'][res['lim']]:g} us"
                if res["pri_mode"] == "switch" else "")
    mode = (f"after coherent integration (N = {res['n']}{lim_note})"
            if sim["integrated"] else "single pulse")
    ax3.set_title(f"Matched filter OUTPUT (MF BW = {res['b_rx']/1e6:g} MHz),"
                  f" {mode}\n"
                  f"peak SNR {sim['snr_db']:.1f} dB (theory), "
                  f"{sim['measured_peak_db']:.1f} dB (this noise sample), "
                  f"zoom +/-{sim['zoom_km']*1000:.0f} m", fontsize=9)
    ax3.set_xlabel("range (km)", fontsize=8)
    ax3.set_xlim(res["target_km"] - sim["zoom_km"],
                 res["target_km"] + sim["zoom_km"])

    for ax in (ax2, ax3):
        ax.set_ylabel("power re noise (dB)", fontsize=8)
        ax.axvline(res["target_km"], color="green", lw=0.8, alpha=0.6)
        top = max(10.0, np.max(ax.lines[1].get_ydata()) + 6)
        ax.set_ylim(-30, top)
        ax.legend(loc="upper right", fontsize=7)
    ax3.set_ylim(-30, max(ax3.get_ylim()[1], sim["thr_db"] + 6))
    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    fig.tight_layout()


def doppler_window(name, n):
    """Slow-time weighting window (numpy array of n weights)."""
    if name == "Hamming":
        return np.hamming(n)
    if name == "Hann":
        return np.hanning(n + 2)[1:-1]
    if name == "Blackman":
        return np.blackman(n + 2)[1:-1]
    return np.ones(n)


DOPPLER_WINDOWS = ("Rectangular", "Hamming", "Hann", "Blackman")


def simulate_doppler(res, window="Hamming", seed=1, grid=8192):
    """
    Doppler filter bank and the target's Doppler line for the limiting dwell.

    Slow time: n pulses, one complex sample per PRI at the target range gate,
        x[k] = a exp(j 2 pi f_app k PRI) + complex white noise,
    f_app = true Doppler folded into +-PRF/2 (aliasing included). The n-point
    weighted FFT (zero padded to the next power of two >= n) is the bank of
    nfft Doppler filters; bin spacing PRF / nfft. The amplitude is calibrated
    so that the noise-free filter output at the target Doppler / mean noise
    power = integrated SNR from the radar equation (res["snr_int_tgt_db"]).
    """
    d = res["dwells"][res["lim"]]
    n = d["n"]
    if n < 2:
        raise ValueError("Only 1 pulse per dwell: there is no Doppler "
                         "processing (Doppler resolution = PRF).")
    prf = d["prf_hz"]
    f_app = d["dop_app_hz"]
    nfft = d["n_fft"]
    m = nfft * max(1, -(-grid // nfft))          # padded length, multiple
    w = doppler_window(window, n)
    sw, sw2 = float(np.sum(w)), float(np.sum(w ** 2))
    pl_db = 10 * math.log10(n * sw2 / sw ** 2)    # window processing loss

    freq = np.fft.fftshift(np.fft.fftfreq(m, 1 / prf))
    wmag = np.abs(np.fft.fft(w, m)) / sw          # filter shape, 1 at peak
    ks = list(range(nfft))
    if nfft > 64:                                 # draw only the neighbours
        centre = int(round((f_app % prf) / prf * nfft)) % nfft
        ks = sorted({(centre + o) % nfft for o in range(-8, 9)})
    bank = []
    for k in ks:
        h = np.roll(wmag, k * (m // nfft))        # filter centred at k*PRF/nfft
        bank.append(np.fft.fftshift(h))
    centres = [((k if k < nfft / 2 else k - nfft) * prf / nfft) for k in ks]

    snr_db = res["snr_int_tgt_db"]
    jit_db = d.get("jit_loss_db", 0.0)
    # amplitude for the jitter-free SNR; the jitter phases then lower the peak
    amp = math.sqrt(10 ** ((snr_db + jit_db) / 10) * sw2 / sw ** 2)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / prf
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
    ideal = amp * np.exp(2j * math.pi * f_app * t)
    chain = res["chain"]
    jitter = not chain.ideal
    pri_s, tau = d["pri_us"] * 1e-6, res["tau_tgt_s"]
    clean = ideal * chain.gains(n, pri_s, tau,
                                np.random.default_rng(seed + 7919))

    def spec(x, size):
        return 10 * np.log10(np.maximum(
            np.abs(np.fft.fft(w * x, size)) ** 2 / sw2, 1e-12))

    expected = None
    if jitter:          # mean target spectrum over all jitter realisations
        expected = np.fft.fftshift(10 * np.log10(np.maximum(
            amp ** 2 * chain.spectrum(w, m, pri_s, tau, f_app / prf) / sw2,
            1e-12)))
    bins_total = spec(clean + noise, nfft)
    # every PRI dwell folds the true Doppler into its own +-PRF/2; combining
    # the dwells (PRI Switcher) recovers it inside +-1/(2 GCD(PRI))
    f_true = res["f_dop_hz"]
    span = res["dop_span_hz"]
    dwells = [{"pri_us": dw["pri_us"], "prf": dw["prf_hz"],
               "f_app": dw["dop_app_hz"], "detected": dw["detected"],
               "why": dw["why"]} for dw in res["dwells"]]
    return {
        "multi": res["pri_mode"] == "switch", "dwells": dwells,
        "lim_pri_us": d["pri_us"], "span": span,
        "f_resolved": (f_true + span / 2.0) % span - span / 2.0,
        "f_found": res["meas"]["dop_hz"], "found_state": res["meas"]["dop_state"],
        "window": window, "n": n, "prf": prf, "nfft": nfft, "freq": freq,
        "bank": np.array(bank), "centres": centres,
        "fine_clean": np.fft.fftshift(spec(clean, m)),
        "fine_ideal": np.fft.fftshift(spec(ideal, m)) if jitter else None,
        "fine_expected": expected,
        "jit_db": jit_db, "dphi": d.get("jit_dphi", 0.0),
        "df_eff": d.get("df_eff_hz", d["df_res_hz"]),
        "jit_factor": d.get("jit_res_factor", 1.0),
        "fine_total": np.fft.fftshift(spec(clean + noise, m)),
        "bin_f": np.fft.fftshift(np.fft.fftfreq(nfft, 1 / prf)),
        "bin_total": np.fft.fftshift(bins_total),
        "f_app": f_app, "f_true": res["f_dop_hz"], "aliased": d["dop_aliased"],
        "snr_db": snr_db, "thr_db": res["snr_req_db"],
        "measured_db": float(np.max(bins_total)), "pl_db": pl_db,
        "df_res": d["df_res_hz"], "dv_res": d["dv_res"],
        "wavelength": res["wavelength"],
    }


def draw_doppler(fig, res, sim, zoom=True):
    """Doppler filter bank (top) and target Doppler line (bottom).
    zoom: limit the filter bank plot to a few bins around the target."""
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 1.2]})
    lam, prf = sim["wavelength"], sim["prf"]
    half = prf / 2.0
    f = sim["freq"]
    for h in sim["bank"]:
        ax1.plot(f, 20 * np.log10(np.maximum(h, 1e-6)), color="#9aa5b1",
                 lw=0.7)
    # filter that holds the target
    j = int(np.argmin(np.abs(np.array(sim["centres"]) - sim["f_app"])))
    hj = 20 * np.log10(np.maximum(sim["bank"][j], 1e-6))
    ax1.plot(f, hj, color="#c0392b", lw=1.6,
             label="filter containing the target")
    ax1.axvline(sim["f_app"], color="green", lw=1.0, ls="--",
                label=f"target line {sim['f_app']:+,.0f} Hz")
    ax1.set_ylim(-60, 3)
    ax1.set_ylabel("filter response (dB)", fontsize=8)
    shown = ("all" if len(sim["bank"]) == sim["nfft"]
             else f"nearest {len(sim['bank'])}")
    ax1.set_title(
        f"Doppler filter bank: {sim['nfft']} filters ({shown} drawn), "
        f"{sim['window']} window"
        f"{', zoomed' if zoom else ''}\n"
        f"spacing {prf / sim['nfft']:.1f} Hz, resolution "
        f"{sim['df_res']:.1f} Hz = {sim['dv_res']:.2f} m/s",
        fontsize=9, pad=34)
    ax1.legend(loc="upper right", fontsize=7)

    ax2.plot(f, sim["fine_total"], color="#4a90c2", lw=0.8, alpha=0.9,
             label="signal + noise (filter outputs)")
    jit_on = sim["fine_expected"] is not None
    if jit_on:
        ax2.plot(f, sim["fine_ideal"], color="#7f8c8d", lw=1.0, ls=":",
                 label="signal, no phase noise", zorder=2)
        ax2.plot(f, sim["fine_expected"], color="#8e44ad", lw=1.3,
                 label="signal with phase noise (mean)", zorder=3)
        # jitter view on the filter bank: mean line shape re ideal peak
        ref = float(np.max(sim["fine_ideal"]))
        ax1.plot(f, sim["fine_ideal"] - ref, color="#7f8c8d", lw=1.0,
                 ls=":", label="target line, no phase noise", zorder=4)
        ax1.plot(f, sim["fine_expected"] - ref, color="#8e44ad", lw=1.6,
                 label="target line with phase noise (mean)", zorder=4)
        ax1.legend(loc="upper right", fontsize=7)
    ax2.plot(f, sim["fine_clean"], color="#e67e22", lw=1.5,
             label="signal only" + (" (this phase noise sample)"
                                    if jit_on else ""),
             zorder=3)
    ax2.plot(sim["bin_f"], sim["bin_total"], ".", color="#1f4e79", ms=4,
             label="FFT bins")
    ax2.axhline(0, color="gray", ls="--", lw=1.2,
                label="noise floor (mean power)")
    ax2.axhline(sim["thr_db"], color="#c0392b", ls="--", lw=1.2,
                label=f"required SNR ({sim['thr_db']:.1f} dB)")
    ax2.axvline(sim["f_app"], color="green", lw=0.8, alpha=0.6)
    top = max(10.0, sim["snr_db"] + sim["jit_db"] + 8, sim["thr_db"] + 8)
    ax2.set_ylim(-40 if jit_on else -30, top)
    ax2.set_ylabel("power re noise (dB)", fontsize=8)
    ax2.set_xlabel("Doppler (Hz)", fontsize=8)
    v_true = sim["f_true"] * lam / 2
    dwell_tag = f"this dwell (PRI {sim['lim_pri_us']:g} us): " \
        if sim["multi"] else ""
    note = (f"{dwell_tag}ALIASED: true {sim['f_true']:+,.0f} Hz "
            f"({v_true:+.1f} m/s) folds to {sim['f_app']:+,.0f} Hz"
            if sim["aliased"]
            else f"{dwell_tag}true Doppler {sim['f_true']:+,.0f} Hz "
                 f"({v_true:+.1f} m/s), inside +-PRF/2")
    if sim["multi"]:
        folds = ", ".join(
            f"{dw['f_app']:+,.0f}" + ("" if dw["detected"] else "*")
            for dw in sim["dwells"])
        v_res = sim["f_resolved"] * lam / 2
        if sim["f_found"] is not None and sim["found_state"] == "resolved":
            verdict = "resolved across dwells"
        elif sim["f_found"] is None:
            verdict = "would resolve to (NOT DETECTED)"
        else:
            verdict = f"across dwells ({sim['found_state']})"
        note += (f"\nper dwell [Hz]: {folds}  ->  {verdict}: "
                 f"{sim['f_resolved']:+,.0f} Hz ({v_res:+.1f} m/s), "
                 f"span +-{sim['span'] / 2:,.0f} Hz")
        if any(not dw["detected"] for dw in sim["dwells"]):
            note += "   (* = not detected in that dwell)"
    budget_loss = next((v for name, v, _n in res["budget"]
                        if name.startswith("Doppler window")), None)
    loss_txt = (f"; window loss {sim['pl_db']:.2f} dB here vs "
                f"{budget_loss:.2f} dB in the loss budget"
                if budget_loss is not None else "")
    jit_txt = ""
    if jit_on:
        jit_txt = (f"\nsignal chain phase noise: pulse-to-pulse "
                   f"{math.degrees(sim['dphi']):.3g} deg rms, loss "
                   f"{sim['jit_db']:.2f} dB, line width x"
                   f"{sim['jit_factor']:.2f} ({sim['df_eff']:,.1f} Hz)")
    ax2.set_title(
        f"Target Doppler line, N = {sim['n']}, PRF {prf:,.0f} Hz\n{note}\n"
        f"peak SNR {sim['snr_db']:.1f} dB (theory), "
        f"{sim['measured_db']:.1f} dB (this noise sample)\n"
        f"{loss_txt.lstrip('; ')}{jit_txt}", fontsize=9)
    ax2.legend(loc="upper right", fontsize=7)
    for ax in (ax1, ax2):
        ax.set_xlim(-half, half)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    if zoom:                                       # +-6 bins around target
        w6 = 6 * prf / sim["nfft"]
        lo = max(-half, sim["f_app"] - w6)
        hi = min(half, sim["f_app"] + w6)
        ax1.set_xlim(lo, hi)
        ax2.axvspan(lo, hi, color="#c0392b", alpha=0.06, lw=0)
    sec = ax1.secondary_xaxis(
        "top", functions=(lambda x: x * lam / 2, lambda v: v * 2 / lam))
    sec.set_xlabel("radial velocity (m/s)", fontsize=8)
    sec.tick_params(labelsize=7)
    fig.tight_layout()


# =================================================================
# SIGNAL CHAIN VIEW (numpy + matplotlib)
# =================================================================

CHAIN_COLORS = ["#2c7fb8", "#e67e22", "#8e44ad", "#c0392b", "#16a085",
                "#7f8c8d", "#d4ac0d"]


def _chain_block(ax, x, y, w, h, title, body, color):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                fc=color, ec="#34495e", lw=1.0, alpha=0.9))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top",
            fontsize=8.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top",
            fontsize=7, linespacing=1.25)


def _chain_arrow(ax, p0, p1, text="", ls="-", color="#34495e"):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2, ls=ls))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.08, text,
                ha="center", va="bottom", fontsize=6.5, color=color)


def _mixer(ax, x, y, label="", r=0.22):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, fc="white", ec="#34495e", lw=1.0))
    ax.text(x, y, "X", ha="center", va="center", fontsize=11, weight="bold")
    if label:
        ax.text(x + r + 0.04, y + r, label, fontsize=6.5, color="#555555")


def draw_chain(fig, res):
    """Block diagram of the signal chain and the effect of every block vs
    target range (limiting dwell: its N and PRI)."""
    fig.clear()
    ch = res["chain"]
    sp = ch.spec
    d = res["dwells"][res["lim"]]
    n, pri = d["n"], d["pri_us"] * 1e-6
    comps = d["jit_comps"]

    def contrib(names):
        rows = [c for c in comps if any(k in c[0] for k in names)]
        if not rows:
            return ""
        dphi = math.degrees(math.sqrt(sum(c[2] ** 2 for c in rows)))
        fl = [c[3] for c in rows if c[3] is not None]
        txt = f"Δφ pulse-to-pulse {dphi:.3g} deg rms"
        if fl:
            txt += f"\nfloor {10 * math.log10(sum(10 ** (v / 10) for v in fl)):.1f} dBc"
        return txt

    gs = fig.add_gridspec(4, 2, height_ratios=[1.35, 0.16, 1.0, 1.0],
                          hspace=0.5, wspace=0.2)
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.25 if ch.mag else 0, 3.45)
    ax.axis("off")
    osc = "L(f): plot (c)"
    tgt = f"Target\n{res['target_km']:g} km\ntau {res['tau_tgt_s']*1e6:,.1f} us"
    ax.text(8.9, 2.65, tgt, ha="center", va="center", fontsize=8)
    if not ch.mag:
        _chain_block(ax, 0.1, 1.95, 2.0, 1.4, "Source (exciter)",
                     osc + "\n" + contrib(["Source"]), "#d6eaf8")
        amp = sp["amp"]
        if amp == "None (source direct)":
            _chain_block(ax, 2.7, 1.95, 2.0, 1.4, "No amplifier",
                         "source drives\nthe antenna", "#eeeeee")
        else:
            _chain_block(ax, 2.7, 1.95, 2.0, 1.4, amp,
                         f"{sp['amp_push']:g} deg/% x {sp['amp_supply']:g} %"
                         f"\n+ {sp['amp_add']:g} deg rms\n"
                         + contrib([amp]), "#fdebd0")
        _chain_arrow(ax, (2.1, 2.65), (2.7, 2.65))
        _chain_arrow(ax, (4.7, 2.65), (5.3, 2.65))
        if sp["rx_lo"] == "shared":
            ax.plot([1.1, 1.1, 5.3], [1.95, 0.55, 0.55], color="#2c7fb8",
                    lw=1.2)
            _chain_arrow(ax, (5.3, 0.55), (5.55, 0.55), color="#2c7fb8")
            ax.text(2.9, 0.62, "LO = same oscillator (range correlation)",
                    fontsize=7, color="#2c7fb8")
        else:
            _chain_block(ax, 2.7, 0.05, 2.0, 1.3, "RX LO (independent)",
                         "L(f): plot (c)\n" + contrib(["RX LO"]),
                         "#d5f5e3")
            _chain_arrow(ax, (4.7, 0.55), (5.55, 0.55))
    else:
        # Magnetron: TX sample -> M1 (x STALO) -> IF -> locks the COHO;
        # echo -> M2 (x STALO) -> IF -> M3 (x COHO) -> I/Q baseband
        def dphi_of(name):
            rows = [c for c in comps if c[0] == name]
            return math.degrees(rows[0][2]) if rows else None

        mode = sp["coho_mode"]
        if mode == "none":
            mag_txt = (f"P2P {sp['mag_fjit_khz']:g} kHz, intrapulse "
                       f"{sp['mag_chirp_khz']:g} kHz\nrandom start phase\n"
                       "every pulse")
        else:
            e1 = dphi_of("Magnetron P2P freq. jitter")
            e2 = dphi_of("Magnetron intrapulse freq. jitter")
            fl = contrib(["Magnetron"]).split("\n")
            mm = [c for c in comps if c[0].startswith(
                "Magnetron P2P freq. mismatch")]
            mm_txt = (f"MF mismatch {mm[0][1]:.2f} dB" if mm
                      else "MF = TX replica")
            mag_txt = (f"P2P {sp['mag_fjit_khz']:.3g} kHz: Δφ {e1:.3g}°\n"
                       f"intra {sp['mag_chirp_khz']:.3g} kHz: Δφ {e2:.3g}°\n"
                       f"{mm_txt}, PW {res['pw_us']:g} us"
                       + (f"\n{fl[1]}" if len(fl) > 1 else ""))
        _chain_block(ax, 0.1, 1.9, 2.35, 1.52, "Magnetron", mag_txt,
                     "#fadbd8")
        _chain_arrow(ax, (2.4, 2.65), (5.3, 2.65), "RF")
        _chain_block(ax, 2.6, 1.25, 2.2, 1.15, "STALO",
                     osc + "\n" + contrib(["STALO"]).split("\n")[0], "#d6eaf8")
        _mixer(ax, 5.8, 1.5, "M2")
        _chain_arrow(ax, (5.8, 1.95), (5.8, 1.7))
        _chain_arrow(ax, (4.8, 1.55), (5.58, 1.52))
        _chain_arrow(ax, (5.8, 1.28), (5.8, 0.8), "  IF", color="#34495e")
        if sp["coho"]:
            _mixer(ax, 1.2, 1.5)
            ax.text(0.92, 1.5, "M1", fontsize=6.5, color="#555555",
                    ha="right", va="center")
            _chain_arrow(ax, (1.2, 1.9), (1.2, 1.72), ls="--")
            ax.text(1.26, 1.8, "TX sample (RF)", fontsize=6.5, va="center")
            _chain_arrow(ax, (2.6, 1.55), (1.42, 1.5))
            _chain_arrow(ax, (1.2, 1.28), (1.2, 1.12))
            if mode == "analog":
                ax.text(1.42, 1.2, "IF sample: locks COHO", fontsize=6.5,
                        va="center")
                _chain_block(ax, 0.1, -0.2, 3.4, 1.25, "Analog COHO (IF)",
                             f"locked at the "
                             f"{LOCK_POINTS[sp['lock_point']].lower()}, "
                             f"residual {sp['mag_lock']:g} deg\n"
                             + contrib(["COHO"]), "#fdebd0")
                _chain_arrow(ax, (3.5, 0.55), (5.55, 0.55), "IF reference")
            else:
                ax.text(1.42, 1.2, "IF sample -> ADC", fontsize=6.5,
                        va="center")
                _chain_block(ax, 0.1, -0.2, 3.4, 1.25,
                             "Digital COHO (ADC + burst analysis)",
                             f"{DIG_CORR[sp['dig_corr']]}\n"
                             f"meas. error {sp['dig_err']:g} deg, phase memory "
                             + ("on" if sp["dig_memory"] else "off") + "\n"
                             + contrib(["COHO"]).split("\n")[0], "#e8f8f5")
                _chain_arrow(ax, (3.5, 0.55), (5.55, 0.55),
                             "phase / replica per pulse")
        else:
            ax.text(1.8, 0.6, "NO COHO: phase of each pulse unknown",
                    ha="center", fontsize=7.5, color="#c0392b")
        _mixer(ax, 5.8, 0.55, "M3")
        ax.text(5.0, 0.12, "ADC, digital I/Q + correction"
                if mode == "digital" else "I/Q detection",
                fontsize=6.5, ha="center")
    _chain_block(ax, 5.3, 1.95, 1.6, 1.4, "Duplexer +\nantenna", "", "#eeeeee")
    _chain_arrow(ax, (6.9, 2.8), (8.2, 2.8), "TX")
    _chain_arrow(ax, (8.2, 2.45), (6.9, 2.45), "echo", ls="--")
    if not ch.mag:
        _mixer(ax, 5.8, 0.55)
        _chain_arrow(ax, (6.1, 1.95), (5.85, 0.82))
    _chain_block(ax, 6.7, 0.05, 3.1, 1.3, "Matched filter + Doppler FFT",
                 f"N = {n}, PRI {d['pri_us']:g} us\nloss {d['jit_loss_db']:.2f}"
                 f" dB, floor "
                 + ("-" if d["jit_floor_dbc"] is None
                    else f"{d['jit_floor_dbc']:.1f} dBc")
                 + f"\nline width x{d['jit_res_factor']:.2f}", "#e8daef")
    _chain_arrow(ax, (6.05, 0.55), (6.7, 0.55))
    ax.set_title(CHAIN_TYPES[sp["type"]] + "  -  phase noise of every block "
                 "adds; effects at the target range", fontsize=9)

    # ---- (a) loss and (b) Doppler floor vs range, per block ----
    r_max = max(2.0 * res["target_km"], 1.2 * res["r_coh"] / 1000.0, 5.0)
    rk = np.linspace(r_max / 60, r_max, 60)
    taus = 2 * rk * 1000.0 / C
    a1 = fig.add_subplot(gs[2, 0])
    a2 = fig.add_subplot(gs[2, 1])
    a3 = fig.add_subplot(gs[3, 0])
    a4 = fig.add_subplot(gs[3, 1])
    sets = [([c], c[0], CHAIN_COLORS[i % len(CHAIN_COLORS)])
            for i, c in enumerate(ch.comps)]
    sets.append((ch.comps, "TOTAL", "black"))
    handles = []
    for cs, name, col in sets:
        lw, ls = (2.2, "-") if name == "TOTAL" else (1.4, "--")
        h, = a1.plot(rk, [ch.loss_db(n, pri, t, cs) for t in taus],
                     color=col, lw=lw, ls=ls, label=name)
        handles.append(h)
        fl = [doppler_floor_dbc(ch, n, pri, t, cs) for t in taus]
        if any(v is not None for v in fl):
            a2.plot(rk, [np.nan if v is None else v for v in fl], color=col,
                    lw=lw, ls=ls)
    # mark the values at the target range on the TOTAL curves
    a1.plot(res["target_km"], d["jit_loss_db"], "o", color="black", ms=6,
            zorder=5)
    a1.annotate(f" {d['jit_loss_db']:.2f} dB", (res["target_km"],
                d["jit_loss_db"]), fontsize=8.5, weight="bold",
                xytext=(8, -16), textcoords="offset points")
    if d["jit_floor_dbc"] is not None:
        a2.plot(res["target_km"], d["jit_floor_dbc"], "o", color="black",
                ms=6, zorder=5)
        a2.annotate(f" {d['jit_floor_dbc']:.1f} dBc (TOTAL)",
                    (res["target_km"], d["jit_floor_dbc"]), fontsize=8.5,
                    weight="bold", xytext=(8, -16), textcoords="offset points")
    a1.set_title("(a) Coherent loss of the target (dB)", fontsize=10)
    a2.set_title("(b) Doppler phase-noise floor (dBc)", fontsize=10)
    ru_km = C * pri / 2000.0
    show_ru = (any(k == "trip" for _n, k, _p in ch.comps)
               and ru_km < r_max)
    for a in (a1, a2):
        a.axvline(res["target_km"], color="green", lw=1.2, alpha=0.8)
        if show_ru:
            a.axvline(ru_km, color="#c0392b", lw=1.2, ls=":", alpha=0.9)
        a.set_xlabel(f"target range (km)  [N = {n}, PRI {d['pri_us']:g} us]",
                     fontsize=8.5)
        a.tick_params(labelsize=8)
        a.grid(alpha=0.3)
    if show_ru:
        a1.text(ru_km, 0.5, " Ru: 2nd trip incoherent", rotation=90,
                transform=a1.get_xaxis_transform(), fontsize=7.5,
                color="#c0392b", va="center")
    lax = fig.add_subplot(gs[1, :])
    lax.axis("off")
    lax.legend(handles=handles, loc="center", ncol=min(3, len(handles)),
               fontsize=7.5, frameon=False, handlelength=2.2,
               columnspacing=1.2)

    # ---- (c) oscillator phase noise inputs ----
    tau_t = res["tau_tgt_s"]
    f = np.geomspace(10.0, 1e8, 600)
    pw = res["pw_us"] * 1e-6
    shared_names = {"COHO (IF)"} | ({"STALO", "Source"}
                                    if ch.mag or sp["rx_lo"] == "shared"
                                    else set())
    for i, (label, pn) in enumerate(ch.oscillators()):
        col = ["#2c7fb8", "#16a085", "#e67e22"][i % 3]
        a3.semilogx(f, pn.l_dbc(f), color=col, lw=2, label=f"{label} L(f)")
        a3.plot(pn.f_hz, pn.l_db, "o", color=col, ms=5)
        if label in shared_names:
            # mean of 4 sin^2(pi f tau): (2 pi f tau)^2 below, 2 above
            eff = pn.l_dbc(f) + 10 * np.log10(
                np.maximum(np.minimum((2 * np.pi * f * tau_t) ** 2, 2.0),
                           1e-30))
            a3.semilogx(f, eff, color=col, lw=1.0, ls="--", alpha=0.8,
                        label=f"{label}: left after range correlation")
    for x, txt in ((1 / (2 * pri), "PRF/2"), (1 / pw, "1/PW")):
        a3.axvline(x, color="gray", lw=1, ls=":")
        a3.text(x, 0.98, " " + txt, transform=a3.get_xaxis_transform(),
                fontsize=7.5, color="gray", va="top")
    a3.set_ylim(-190, -30)
    a3.set_xlim(10, 1e8)
    a3.set_title("(c) Oscillator L(f)  (points = inputs)", fontsize=10)
    a3.set_xlabel("offset from carrier (Hz)", fontsize=8.5)
    a3.set_ylabel("dBc/Hz", fontsize=8.5)
    a3.tick_params(labelsize=8)
    a3.grid(alpha=0.3, which="both")
    a3.legend(fontsize=7, loc="upper right", framealpha=0.85)

    # ---- (d) expected Doppler spectrum at the target range ----
    size = 4096
    win = blackman_harris(n) if n >= 8 else np.ones(n)
    s_id = ch.spectrum(win, size, pri, tau_t, comps=[])
    s_ch = ch.spectrum(win, size, pri, tau_t)
    fx = (np.arange(size) - size // 2) / size / pri / 1e3
    pk = max(s_id.max(), 1e-30)
    for sv, col, lw, lab in ((s_id, "#95a5a6", 1.2,
                              "ideal (window sidelobes only)"),
                             (s_ch, "black", 1.6, "with signal chain")):
        a4.plot(fx, 10 * np.log10(np.maximum(np.fft.fftshift(sv), 1e-30)
                                  / pk), color=col, lw=lw, label=lab)
    if d["jit_floor_dbc"] is not None:
        a4.axhline(d["jit_floor_dbc"], color="black", lw=1.2, ls=":",
                   label=f"floor {d['jit_floor_dbc']:.1f} dBc = (b) TOTAL "
                   "at target")
        g_khz = 8.0 / (n * pri) / 1e3            # +-8 bins: not in the floor
        a4.axvspan(-g_khz, g_khz, color="#f4d03f", alpha=0.18, lw=0,
                   label="+-8 bins: excluded from the floor value")
    a4.set_ylim(-110, 5)
    a4.set_xlim(fx[0], fx[-1])
    a4.set_title("(d) Doppler spectrum at the target range", fontsize=10)
    a4.set_xlabel("offset from the echo line (kHz), Blackman-Harris",
                  fontsize=8.5)
    a4.set_ylabel("dB re peak", fontsize=8.5)
    a4.tick_params(labelsize=8)
    a4.grid(alpha=0.3)
    a4.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.96, bottom=0.06)


# =================================================================
# 2D GEOMETRY VIEW (numpy)
# =================================================================

ANT_SPAN_DB = 40.0       # dynamic range of the antenna lobe drawing
RCS_SPAN_DB = 30.0       # dynamic range of the RCS lobe drawing
_GEO_CACHE = {}


def _cache_put(key, value):
    if len(_GEO_CACHE) > 24:
        _GEO_CACHE.clear()
    _GEO_CACHE[key] = value


def off_boresight_np(az, el, ant_az, ant_el):
    """Vectorised off_boresight(). Returns (az_off, el_off) in degrees."""
    a, e = np.radians(az), np.radians(el)
    pa, pe = math.radians(ant_az), math.radians(ant_el)
    vx = np.cos(e) * np.cos(a)
    vy = np.cos(e) * np.sin(a)
    vz = np.sin(e) + 0 * a
    x1 = vx * math.cos(pa) + vy * math.sin(pa)
    y1 = -vx * math.sin(pa) + vy * math.cos(pa)
    x2 = x1 * math.cos(pe) + vz * math.sin(pe)
    z2 = -x1 * math.sin(pe) + vz * math.cos(pe)
    return (np.degrees(np.arctan2(y1, x2)),
            np.degrees(np.arctan2(z2, np.hypot(x2, y1))))


def antenna_cuts(tables, freq_ghz, step=0.25):
    """Principal-plane cuts on a fine grid, cached per table and frequency.
    Returns (angles, az_cut, el_cut, peak)."""
    tbl = tables["antenna_gain"]
    key = ("gain", round(freq_ghz, 6))
    hit = _GEO_CACHE.get(key)
    if hit is None or hit[0] is not tbl:
        angles = np.arange(0.0, 180.0 + step, step)
        azc = np.array([grid_value(tbl, freq_ghz, a, 0)[0] for a in angles])
        elc = np.array([grid_value(tbl, freq_ghz, a, 1)[0] for a in angles])
        hit = (tbl, angles, azc, elc, max(azc[0], elc[0]))
        _cache_put(key, hit)
    return hit[1:]


def gain_map(tables, freq_ghz, az_off, el_off):
    """Vectorised antenna_gain(): az cut + el cut - peak. Returns (dBi, peak)."""
    angles, azc, elc, peak = antenna_cuts(tables, freq_ghz)
    g = (np.interp(np.abs(az_off), angles, azc)
         + np.interp(np.abs(el_off), angles, elc) - peak)
    return np.maximum(g, peak - 60.0), peak


def rcs_curve(tables, freq_ghz, pol, step=0.5):
    """RCS (dBsm) vs aspect 0..180 deg, cached. Returns (angles, dBsm)."""
    tbl = tables["rcs"]
    key = ("rcs", round(freq_ghz, 6), pol)
    hit = _GEO_CACHE.get(key)
    if hit is None or hit[0] is not tbl:
        angles = np.arange(0.0, 180.0 + step, step)
        vals = np.array([10 * math.log10(rcs_value(tables, freq_ghz, a, pol)[0])
                         for a in angles])
        hit = (tbl, angles, vals)
        _cache_put(key, hit)
    return hit[1], hit[2]


class GeometryView:
    """Four live-updating 2D plots. Artists are created once; update() only
    changes their data, so dragging a slider stays smooth."""

    BLUE, GREEN, ORANGE, GRAY = "#2e86c1", "#27ae60", "#d35400", "#7f8c8d"

    def __init__(self, fig):
        self.fig = fig
        gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.42,
                              wspace=0.30, left=0.09, right=0.97,
                              top=0.91, bottom=0.08)
        self.ax_plan = fig.add_subplot(gs[0, 0])
        self.ax_side = fig.add_subplot(gs[0, 1])
        self.ax_gain = fig.add_subplot(gs[1, 0])
        self.ax_rcs = fig.add_subplot(gs[1, 1])
        self.status = fig.text(0.5, 0.975, "", ha="center", va="center",
                               fontsize=9, fontweight="bold")
        self.last = {}
        self._build_plan()
        self._build_side()
        self._build_gain()
        self._build_rcs()

    # ---------------- construction ----------------
    def _style(self, ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    def _build_plan(self):
        ax = self.ax_plan
        self._style(ax, "Plan view (top-down, north up)", "East (km)",
                    "North (km)")
        ax.set_aspect("equal", adjustable="box")
        self.p_lobe = ax.fill([0], [0], color=self.BLUE, alpha=0.25, lw=0)[0]
        self.p_lobe_ln, = ax.plot([], [], color=self.BLUE, lw=1.2)
        self.p_bore, = ax.plot([], [], color=self.BLUE, ls="--", lw=1)
        self.p_los, = ax.plot([], [], color=self.GRAY, ls=":", lw=1)
        self.p_rcs = ax.fill([0], [0], color=self.ORANGE, alpha=0.25, lw=0)[0]
        self.p_rcs_ln, = ax.plot([], [], color=self.ORANGE, lw=1.2)
        self.p_rcs_pt, = ax.plot([], [], "o", color=self.ORANGE, mfc="white",
                                 ms=5, zorder=5)
        self.p_tgt, = ax.plot([], [], "D", color=self.ORANGE, ms=6, zorder=6)
        ax.plot([0], [0], "^", color="black", ms=8, zorder=6)
        ax.annotate("Radar", (0, 0), xytext=(6, -12), textcoords="offset points",
                    fontsize=7)
        self.p_head = ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=self.ORANGE, lw=1.5),
            zorder=6)
        self.p_info = ax.text(
            0.02, 0.02, "", transform=ax.transAxes, fontsize=6.5, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.85))
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=self.BLUE, alpha=0.4, label="antenna gain pattern"),
            Patch(color=self.ORANGE, alpha=0.4, label="target RCS pattern"),
        ], loc="upper right", fontsize=6.5)

    def _build_side(self):
        ax = self.ax_side
        self._style(ax, "Side view (vertical plane through target)",
                    "Range along target azimuth (km)", "Height (km)")
        ax.set_aspect("equal", adjustable="box")
        ax.axhspan(-1e6, 0, color="#f2f3f4", zorder=0)
        ax.axhline(0, color="#999999", lw=0.8, zorder=1)
        self.s_lobe = ax.fill([0], [0], color=self.BLUE, alpha=0.25, lw=0)[0]
        self.s_lobe_ln, = ax.plot([], [], color=self.BLUE, lw=1.2)
        self.s_bore, = ax.plot([], [], color=self.BLUE, ls="--", lw=1)
        self.s_los, = ax.plot([], [], color=self.GRAY, ls=":", lw=1)
        self.s_tgt, = ax.plot([], [], "D", color=self.ORANGE, ms=6, zorder=6)
        ax.plot([0], [0], "^", color="black", ms=8, zorder=6)
        self.s_info = ax.text(
            0.02, 0.97, "", transform=ax.transAxes, fontsize=6.5, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.85))

    def _build_gain(self):
        ax = self.ax_gain
        self._style(ax, "Antenna gain cuts", "off-boresight angle (deg)",
                    "gain (dBi)")
        self.g_az, = ax.plot([], [], color=self.BLUE, lw=1.4, label="azimuth cut")
        self.g_el, = ax.plot([], [], color=self.GREEN, lw=1.4, label="elevation cut")
        self.g_pt_az, = ax.plot([], [], "o", color=self.BLUE, mfc="white",
                                ms=6, zorder=5, clip_on=False)
        self.g_pt_el, = ax.plot([], [], "o", color=self.GREEN, mfc="white",
                                ms=6, zorder=5, clip_on=False)
        self.g_txt = ax.text(0.97, 0.05, "", transform=ax.transAxes,
                             fontsize=7, ha="right", va="bottom",
                             bbox=dict(boxstyle="round", fc="white",
                                       ec="#cccccc", alpha=0.85))
        self.g_leg = ax.legend(loc="upper right", fontsize=6.5)

    def _build_rcs(self):
        ax = self.ax_rcs
        self._style(ax, "Target RCS vs aspect", "aspect angle (deg)  "
                    "0 = nose-on, 180 = tail-on", "RCS (dBsm)")
        self.r_ln, = ax.plot([], [], color=self.ORANGE, lw=1.4)
        self.r_pt, = ax.plot([], [], "o", color=self.ORANGE, mfc="white",
                             ms=6, zorder=5, clip_on=False)
        self.r_txt = ax.text(0.97, 0.05, "", transform=ax.transAxes,
                             fontsize=7, ha="right", va="bottom",
                             bbox=dict(boxstyle="round", fc="white",
                                       ec="#cccccc", alpha=0.85))
        ax.set_xlim(-4, 184)

    # ---------------- live update ----------------
    @staticmethod
    def _poly(artist, xs, ys):
        artist.set_xy(np.column_stack([xs, ys]))

    def update(self, res, tables):
        f = res["freq_ghz"]
        R = res["target_km"]
        el = res["tgt_el"]
        az_t = res["tgt_az"]
        ant_az, ant_el = res["ant_az"], res["ant_el"]
        has_pat = res["gain_source"].startswith("table")
        rcs_tab = res["rcs_source"].startswith("table")
        ground = R * math.cos(math.radians(el))
        height = R * math.sin(math.radians(el))
        sig_db = 10 * math.log10(res["rcs_m2"])

        self.status.set_text(
            f"{res['status_text']}\nmargin {res['margin_db']:+.2f} dB at "
            f"{R:,.2f} km")
        self.status.set_color(STATUS_COLOR[res["status_kind"]])

        info_ant = (f"Antenna az {ant_az:.1f}, el {ant_el:.1f} deg\n"
                    f"gain toward target {res['gain_dbi']:.1f} dBi "
                    f"(peak {res['gain_peak_dbi']:.1f})\n"
                    f"off-boresight {res['tot_off']:.1f} deg")
        info_tgt = (f"Target R {R:.1f} km, az {az_t:.1f}, el {el:.1f} deg\n"
                    f"heading {res['tgt_heading']:.0f} deg, "
                    f"speed {res['speed_ms']:g} m/s\n"
                    f"aspect {res['aspect']:.0f} deg, "
                    f"radial {res['radial_ms']:+.1f} m/s\n"
                    f"RCS {res['rcs_m2']:.3g} m2 ({sig_db:.1f} dBsm)")

        # ================= plan view =================
        Rp = max(ground, 0.1 * R)
        phi = np.arange(0.0, 361.0, 1.0)                # compass azimuth
        sin_p, cos_p = np.sin(np.radians(phi)), np.cos(np.radians(phi))
        sin_t = math.sin(math.radians(az_t))
        cos_t = math.cos(math.radians(az_t))
        xt, yt = ground * sin_t, ground * cos_t

        for art in (self.p_lobe, self.p_lobe_ln):
            art.set_visible(has_pat)
        if has_pat:
            a_off, e_off = off_boresight_np(phi, el, ant_az, ant_el)
            g, peak = gain_map(tables, f, a_off, e_off)
            rad = 0.9 * Rp * np.clip((g - (peak - ANT_SPAN_DB)) / ANT_SPAN_DB,
                                     0.0, 1.0)
            self._poly(self.p_lobe, rad * sin_p, rad * cos_p)
            self.p_lobe_ln.set_data(rad * sin_p, rad * cos_p)
            self.last["plan_gain"] = (phi, g)
        sa, ca = (math.sin(math.radians(ant_az)),
                  math.cos(math.radians(ant_az)))
        self.p_bore.set_data([0, 0.9 * Rp * sa], [0, 0.9 * Rp * ca])
        self.p_los.set_data([0, xt], [0, yt])
        self.p_tgt.set_data([xt], [yt])

        Rt = 0.22 * Rp
        asp = np.abs((res["tgt_heading"] - phi + 180.0) % 360.0 - 180.0)
        if rcs_tab:
            ang, vals = rcs_curve(tables, f, res["pol"])
            vmax = float(vals.max())
            sc = lambda v: Rt * np.clip((v - (vmax - RCS_SPAN_DB))
                                        / RCS_SPAN_DB, 0.05, 1.0)
            rr = sc(np.interp(asp, ang, vals))
            r_now = float(sc(sig_db))
            self.last["plan_rcs"] = (phi, np.interp(asp, ang, vals))
        else:
            rr = np.full_like(phi, 0.6 * Rt)
            r_now = 0.6 * Rt
        self._poly(self.p_rcs, xt + rr * sin_p, yt + rr * cos_p)
        self.p_rcs_ln.set_data(xt + rr * sin_p, yt + rr * cos_p)
        # point of the lobe that the radar sees (bearing target -> radar)
        self.p_rcs_pt.set_data([xt - r_now * sin_t], [yt - r_now * cos_t])
        hd = math.radians(res["tgt_heading"])
        self.p_head.xy = (xt + 0.32 * Rp * math.sin(hd),
                          yt + 0.32 * Rp * math.cos(hd))
        self.p_head.set_position((xt, yt))
        self.ax_plan.set_xlim(-1.45 * Rp, 1.45 * Rp)
        self.ax_plan.set_ylim(-1.45 * Rp, 1.45 * Rp)
        self.p_info.set_text(info_ant + "\n" + info_tgt)

        # ================= side view =================
        psi = np.arange(0.0, 361.0, 1.0)         # angle in the vertical plane
        pw = (psi + 180.0) % 360.0 - 180.0
        front = np.abs(pw) <= 90.0
        el_dir = np.where(front, pw, np.sign(pw) * (180.0 - np.abs(pw)))
        az_dir = np.where(front, az_t, az_t + 180.0)
        for art in (self.s_lobe, self.s_lobe_ln, self.s_bore):
            art.set_visible(has_pat)
        if has_pat:
            a_off, e_off = off_boresight_np(az_dir, el_dir, ant_az, ant_el)
            g, peak = gain_map(tables, f, a_off, e_off)
            rad = 0.9 * R * np.clip((g - (peak - ANT_SPAN_DB)) / ANT_SPAN_DB,
                                    0.0, 1.0)
            cx, cy = np.cos(np.radians(psi)), np.sin(np.radians(psi))
            self._poly(self.s_lobe, rad * cx, rad * cy)
            self.s_lobe_ln.set_data(rad * cx, rad * cy)
            k = int(np.argmax(g))
            self.s_bore.set_data([0, 0.9 * R * cx[k]], [0, 0.9 * R * cy[k]])
            self.last["side_gain"] = (psi, g)
        self.s_los.set_data([0, ground], [0, height])
        self.s_tgt.set_data([ground], [height])
        ymax = max(0.6 * R, 1.15 * height)
        ymin = min(-0.4 * R, 1.15 * height)
        self.ax_side.set_xlim(-0.5 * R, max(1.15 * R, 1.15 * ground))
        self.ax_side.set_ylim(ymin, ymax)
        self.s_info.set_text(
            f"target: ground {ground:.1f} km, height {height:.2f} km\n"
            f"lobe = gain in the vertical plane at az {az_t:.1f} deg\n"
            f"dashed blue = beam maximum in this plane")

        # ================= gain cuts =================
        ax = self.ax_gain
        for art in (self.g_az, self.g_el, self.g_pt_az, self.g_pt_el,
                    self.g_leg):
            art.set_visible(has_pat)
        if has_pat:
            angles, azc, elc, peak = antenna_cuts(tables, f)
            xmax = min(180.0, max(12.0, 1.6 * max(abs(res["az_off"]),
                                                  abs(res["el_off"])) + 3.0))
            self.g_az.set_data(angles, azc)
            self.g_el.set_data(angles, elc)
            self.g_pt_az.set_data([abs(res["az_off"])],
                                  [np.interp(abs(res["az_off"]), angles, azc)])
            self.g_pt_el.set_data([abs(res["el_off"])],
                                  [np.interp(abs(res["el_off"]), angles, elc)])
            ax.set_xlim(-0.03 * xmax, xmax)
            ax.set_ylim(peak - 45, peak + 4)
            self.g_txt.set_text(
                f"toward target: az off {abs(res['az_off']):.1f}, "
                f"el off {abs(res['el_off']):.1f} deg\n"
                f"G = az cut + el cut - peak = {res['gain_dbi']:.1f} dBi")
        else:
            ax.set_xlim(0, 12)
            ax.set_ylim(res["gain_dbi"] - 10, res["gain_dbi"] + 5)
            self.g_txt.set_text(f"manual gain {res['gain_dbi']:.1f} dBi "
                                "(constant)\nno pattern: pointing has no effect")

        # ================= RCS vs aspect =================
        ax = self.ax_rcs
        if rcs_tab:
            ang, vals = rcs_curve(tables, f, res["pol"])
            self.r_ln.set_data(ang, vals)
            self.r_ln.set_linestyle("-")
            ax.set_ylim(float(vals.min()) - 3, float(vals.max()) + 3)
        else:
            self.r_ln.set_data([0, 180], [sig_db, sig_db])
            self.r_ln.set_linestyle("--")
            ax.set_ylim(sig_db - 10, sig_db + 10)
        self.r_pt.set_data([res["aspect"]], [sig_db])
        self.r_txt.set_text(
            f"aspect {res['aspect']:.1f} deg, pol {res['pol']}\n"
            f"RCS {res['rcs_m2']:.3g} m2 = {sig_db:.1f} dBsm"
            + ("" if rcs_tab else "\nmanual (constant)"))

        self.fig.canvas.draw_idle()


# =================================================================
# GUI
# =================================================================

def make_scrollable(parent):
    """Vertical scrollable frame (for the input column)."""
    outer = ttk.Frame(parent)
    canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
    vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    inner = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")

    def on_configure(_):
        canvas.configure(scrollregion=canvas.bbox("all"),
                         width=inner.winfo_reqwidth())

    inner.bind("<Configure>", on_configure)

    def on_wheel(event):
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")
        else:
            canvas.yview_scroll(int(-event.delta / 120), "units")

    def bind_wheel(_):
        canvas.bind_all("<MouseWheel>", on_wheel)
        canvas.bind_all("<Button-4>", on_wheel)
        canvas.bind_all("<Button-5>", on_wheel)

    def unbind_wheel(_):
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    canvas.bind("<Enter>", bind_wheel)
    canvas.bind("<Leave>", unbind_wheel)
    return outer, inner


# =================================================================
# MAX RANGE SWEEP (pure, no GUI)
# =================================================================

# key: (label, unit, default from, default to, log spacing, integer)
SWEEP_PARAMS = {
    "pri":  ("PRI",                  "us",  100.0, 500.0, False, False),
    "pw":   ("Pulse width PW",       "us",  0.2,   20.0,  True,  False),
    "duty": ("Duty cycle (PRI fixed, PW follows)", "%", 1.0, 10.0, False, False),
    "p_dbm": ("Peak power",          "dBm", 50.0,  70.0,  False, False),
    "n":    ("Pulse count N",        "",    8,     128,   True,  True),
    "gain": ("Antenna gain",         "dBi", 30.0,  50.0,  False, False),
    "rcs":  ("RCS",                  "m^2", 0.01,  10.0,  True,  False),
}


def sweep_values(lo, hi, steps, log=False, integer=False):
    """Evenly (or log-evenly) spaced values from lo to hi, inclusive."""
    if steps < 2 or steps > 500:
        raise ValueError("Steps must be between 2 and 500.")
    if hi < lo:
        raise ValueError("'To' must not be smaller than 'From'.")
    if log and lo <= 0:
        raise ValueError("Log spacing needs a positive 'From' value.")
    if log:
        ratio = (hi / lo) ** (1.0 / (steps - 1))
        values = [lo * ratio ** i for i in range(steps)]
    else:
        values = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
    if integer:
        values = sorted({max(1, int(round(v))) for v in values})
    return values


def run_sweep(inp, tables, param, values, match_bw=True):
    """
    Detection range for each value of one parameter, everything else as in
    inp (the normal evaluate() input). Returns a list of dicts
    {value, r_coh, r_single, status, error, range_res (m), dv_res (m/s),
    df_res (Hz), sig_v_tgt / sig_v_thr (m/s), sig_f_tgt / sig_f_thr (Hz)};
    r_* are metres. The resolution / accuracy values are those of the
    limiting dwell (None on error). sig_*_tgt = Doppler accuracy (1 sigma)
    at the SNR of the target, sig_*_thr = at the required SNR.

    param: "pri" | "pw" | "duty" | "p_dbm" | "n" | "gain" | "rcs"
           "duty" = duty cycle in %, PRI stays fixed and PW = duty x PRI
           (shortest PRI if several)
    n, gain and rcs force the manual / pulse-count input path.
    match_bw: with param "pw", set the receiver bandwidth to the signal
              bandwidth of each PW (otherwise B_rx stays as entered and the
              bandwidth-mismatch loss changes with PW).
    r_coh is the coherent detection range of the limiting dwell.
    """
    if param not in SWEEP_PARAMS:
        raise ValueError(f"Unknown parameter {param}.")
    if param == "pri" and inp.get("pri_mode") == "switch":
        raise ValueError("Sweeping the PRI needs PRI Stable mode.")
    rows = []
    for v in values:
        i = dict(inp)
        try:
            if param == "pri":
                i["pri_list"] = [v]
            elif param in ("pw", "duty"):
                i["pw"] = (v if param == "pw"
                           else v / 100.0 * min(i["pri_list"]))
                if match_bw:
                    b_sig, _note = signal_bandwidth_hz(
                        i["pw"], i["mop"], i["mop_param"])
                    i["bw"] = b_sig / 1e6
            elif param == "p_dbm":
                i["p_dbm"] = v
            elif param == "n":
                i["n_pulses"], i["cpi"] = int(v), None
            elif param == "gain":
                i["gain_table"], i["gain_manual"] = False, v
            elif param == "rcs":
                i["rcs_table"], i["rcs_manual"] = False, v
            res = evaluate(i, tables)
            d = min(res["dwells"], key=lambda x: x["int_gain_db"])
            rows.append({"value": v, "r_coh": res["r_coh"],
                         "r_single": res["r_single"],
                         "status": res["status_text"], "error": None,
                         "range_res": res["range_res"],
                         "dv_res": d["dv_res"], "df_res": d["df_res_hz"],
                         "sig_v_tgt": d["sig_v_tgt"],
                         "sig_v_thr": d["sig_v_thr"],
                         "sig_f_tgt": d["sig_f_tgt"],
                         "sig_f_thr": d["sig_f_thr"]})
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            rows.append({"value": v, "r_coh": None, "r_single": None,
                         "status": "", "error": str(exc)})
    return rows


def best_row(rows):
    """Row with the longest coherent detection range (None if none valid)."""
    valid = [r for r in rows if r["r_coh"] is not None]
    return max(valid, key=lambda r: r["r_coh"]) if valid else None


def build_gui():
    table_dir = default_table_dir()
    state = {"tables": None, "table_msgs": [], "suppress": False,
             "pending": None, "sync": False}
    state["tables"], state["table_msgs"] = load_tables(table_dir)

    root = tk.Tk()
    root.title("Radar Detection Range Calculator")
    root.geometry("1500x900")
    root.minsize(1050, 600)

    main = ttk.Frame(root, padding=14)
    main.pack(fill="both", expand=True)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(0, weight=1)

    left_outer, left = make_scrollable(main)
    left_outer.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

    def section(title, row):
        frame = ttk.LabelFrame(left, text=title, padding=10)
        frame.grid(row=row, column=0, sticky="ew", pady=(0 if row == 0 else 10, 0))
        frame.columnconfigure(1, weight=1)
        return frame

    radar_f = section("Radar", 0)
    tx_f = section("Transmitter / signal chain (phase noise)", 1)
    ant_f = section("Antenna", 2)
    tgt_f = section("Target", 3)
    loss_f = section("Losses ('table' = read from frequency table)", 4)

    notebook = ttk.Notebook(main)
    notebook.grid(row=0, column=1, rowspan=2, sticky="nsew")
    out = ttk.Frame(notebook, padding=8)
    mf_tab = ttk.Frame(notebook, padding=8)
    notebook.add(out, text="Results")
    notebook.add(mf_tab, text="Matched filter")
    geo_tab = ttk.Frame(notebook, padding=8)
    notebook.add(geo_tab, text="Geometry (2D)")
    geo_tab.columnconfigure(0, weight=1)
    geo_tab.rowconfigure(0, weight=1)
    max_tab = ttk.Frame(notebook, padding=8)
    notebook.add(max_tab, text="Optimizasyon")
    chain_tab = ttk.Frame(notebook, padding=8)
    notebook.insert(0, chain_tab, text="Signal chain")
    notebook.select(out)
    max_tab.columnconfigure(0, weight=1)
    max_tab.rowconfigure(1, weight=1)

    def tab_is(frame):
        return notebook.select() == str(frame)
    out.columnconfigure(0, weight=1)
    out.rowconfigure(0, weight=1)
    mf_tab.columnconfigure(0, weight=1)
    mf_tab.rowconfigure(1, weight=1)

    mf = {"fig": None, "canvas": None, "res": None, "seed": 1}
    mf_mode = tk.StringVar(value="single")
    mf_win = tk.StringVar(value="Hamming")
    mf_zoom = tk.BooleanVar(value=True)

    def redraw_mf(*_):
        if mf["fig"] is None or mf["res"] is None:
            return
        if not tab_is(mf_tab):
            return                     # draw only when the tab is visible
        try:
            if mf_mode.get() == "doppler":
                sim = simulate_doppler(mf["res"], mf_win.get(),
                                       seed=mf["seed"])
                draw_doppler(mf["fig"], mf["res"], sim, zoom=mf_zoom.get())
            else:
                sim = simulate_mf(mf["res"],
                                  integrated=mf_mode.get() == "integrated",
                                  seed=mf["seed"])
                draw_mf(mf["fig"], mf["res"], sim)
        except ValueError as exc:
            mf["fig"].clear()
            msg = "\n".join(textwrap.wrap(str(exc), 60))
            mf["fig"].text(0.05, 0.9,
                           f"Matched filter view not available:\n\n{msg}",
                           va="top", fontsize=10)
        mf["canvas"].draw_idle()

    def new_noise():
        mf["seed"] += 1
        redraw_mf()

    if HAVE_PLOT:
        ctrl = ttk.Frame(mf_tab)
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Radiobutton(ctrl, text="Single pulse", value="single",
                        variable=mf_mode, command=redraw_mf).pack(side="left")
        ttk.Radiobutton(ctrl, text="Coherent integration (N pulses)",
                        value="integrated", variable=mf_mode,
                        command=redraw_mf).pack(side="left", padx=(10, 0))
        ttk.Radiobutton(ctrl, text="Doppler filters", value="doppler",
                        variable=mf_mode, command=redraw_mf).pack(
            side="left", padx=(10, 0))
        win_box = ttk.Combobox(ctrl, textvariable=mf_win, width=11,
                               state="readonly", values=DOPPLER_WINDOWS)
        win_box.pack(side="left", padx=(4, 0))
        win_box.bind("<<ComboboxSelected>>", redraw_mf)
        ttk.Checkbutton(ctrl, text="Zoom on target", variable=mf_zoom,
                        command=redraw_mf).pack(side="left", padx=(8, 0))
        ttk.Button(ctrl, text="New noise", command=new_noise).pack(
            side="left", padx=(16, 0))
        mf["fig"] = Figure(figsize=(7, 7), dpi=100)
        mf["canvas"] = FigureCanvasTkAgg(mf["fig"], master=mf_tab)
        mf["canvas"].get_tk_widget().grid(row=1, column=0, sticky="nsew")
        bar = ttk.Frame(mf_tab)
        bar.grid(row=2, column=0, sticky="ew")
        NavigationToolbar2Tk(mf["canvas"], bar)
    else:
        ttk.Label(
            mf_tab, justify="left",
            text="The matched filter view needs numpy and matplotlib:\n\n"
                 "    pip install numpy matplotlib\n\n"
                 "Then restart the program.").grid(row=0, column=0, sticky="nw")

    # ---- 2D geometry tab ----
    geo = {"view": None, "res": None}

    def redraw_geo(*_):
        if geo["view"] is None or geo["res"] is None or not tab_is(geo_tab):
            return
        try:
            geo["view"].update(geo["res"], state["tables"])
        except (ValueError, KeyError) as exc:
            geo["view"].status.set_text(f"Geometry view error: {exc}")
            geo["view"].status.set_color("#c0392b")
            geo["view"].fig.canvas.draw_idle()

    if HAVE_PLOT:
        geo_fig = Figure(figsize=(8, 7.5), dpi=100)
        geo_canvas = FigureCanvasTkAgg(geo_fig, master=geo_tab)
        geo_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        geo_bar = ttk.Frame(geo_tab)
        geo_bar.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(geo_canvas, geo_bar)
        geo["view"] = GeometryView(geo_fig)
    else:
        ttk.Label(
            geo_tab, justify="left",
            text="The 2D geometry view needs numpy and matplotlib:\n\n"
                 "    pip install numpy matplotlib\n\n"
                 "Then restart the program.").grid(row=0, column=0, sticky="nw")

    # ---- Max Range tab: sweep one parameter, find the longest range ----
    sw_names = {p[0]: key for key, p in SWEEP_PARAMS.items()}
    sw_param = tk.StringVar(value=SWEEP_PARAMS["pri"][0])
    sw_lo, sw_hi = tk.StringVar(), tk.StringVar()
    sw_steps = tk.StringVar(value="25")
    sw_log = tk.BooleanVar(value=False)
    sw_match = tk.BooleanVar(value=True)
    sw_summary = tk.StringVar(
        value="Choose a parameter and a range, then press Run sweep.")
    sw = {"fig": None, "canvas": None, "rows": None, "unit_label": ""}

    def sw_key():
        return sw_names[sw_param.get()]

    def sw_defaults(*_):
        _label, unit, lo, hi, log, _int = SWEEP_PARAMS[sw_key()]
        sw_lo.set(f"{lo:g}")
        sw_hi.set(f"{hi:g}")
        sw_log.set(log)
        sw_unit.config(text=unit)
        sw_match_chk.state(["!disabled"] if sw_key() in ("pw", "duty")
                           else ["disabled"])

    def run_max_range(quiet=False):
        key = sw_key()
        label, unit, _lo, _hi, _log, integer = SWEEP_PARAMS[key]
        try:
            lo = float(sw_lo.get().replace(",", "."))
            hi = float(sw_hi.get().replace(",", "."))
            steps = int(float(sw_steps.get().replace(",", ".")))
            values = sweep_values(lo, hi, steps, sw_log.get(), integer)
            rows = run_sweep(gather(), state["tables"], key, values,
                             sw_match.get())
        except ValueError as exc:
            if quiet:
                sw_summary.set(f"Optimizasyon: {exc}")
            else:
                messagebox.showerror("Optimizasyon", str(exc))
            return
        sw["rows"] = rows
        sw["unit_label"] = f"{label} ({unit})" if unit else label
        show_sweep(rows, label, unit)

    def show_sweep(rows, label, unit):
        best = best_row(rows)
        bad = [r for r in rows if r["r_coh"] is None]
        flat = False
        if best is None:
            msg = "No valid point in this range."
        else:
            vals = [r["r_coh"] for r in rows if r["r_coh"] is not None]
            flat = max(vals) - min(vals) < 1e-3 * best["r_coh"]
            if flat:
                msg = (f"Range is {best['r_coh'] / 1000:.3f} km for every "
                       f"value: it does not depend on {label} with the "
                       "current inputs (e.g. PRI with a fixed pulse count "
                       "only changes the CPI time).")
            else:
                at_edge = best is rows[0] or best is rows[-1]
                msg = (f"Max coherent range {best['r_coh'] / 1000:.3f} km "
                       f"at {label} = {best['value']:.6g} {unit}".rstrip()
                       + ("   (at the edge of the range: the optimum may "
                          "lie outside it)" if at_edge else ""))
        if bad:
            msg += (f"   [{len(bad)} invalid point(s), first: "
                    f"{bad[0]['value']:.6g} -> {bad[0]['error']}]")
        sw_summary.set(msg)
        if sw["fig"] is None:
            return
        fig = sw["fig"]
        fig.clear()
        ok = [r for r in rows if r["r_coh"] is not None]
        xs = [r["value"] for r in ok]
        use_log = SWEEP_PARAMS[sw_key()][4] and bool(xs) and min(xs) > 0
        axes = fig.subplots(2, 2).ravel()

        def line(ax, key, scale, **kw):
            pts = [(r["value"], r[key] * scale) for r in ok
                   if r.get(key) is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], **kw)
            return bool(pts)

        a_rng, a_rr, a_dr, a_da = axes
        line(a_rng, "r_coh", 1e-3, marker="o", ms=3, color="#2c7fb8",
             label="Coherent (limiting dwell)")
        line(a_rng, "r_single", 1e-3, marker="s", ms=3, ls="--",
             color="#e67e22", label="Single pulse")
        if best is not None:
            a_rng.plot([best["value"]], [best["r_coh"] / 1000], "*", ms=14,
                       color="#c0392b", label="Maximum")
        a_rng.set_title("Detection range", fontsize=10)
        a_rng.set_ylabel("km")
        line(a_rr, "range_res", 1.0, marker="o", ms=3, color="#2c7fb8")
        a_rr.set_title("Range resolution", fontsize=10)
        a_rr.set_ylabel("m")
        line(a_dr, "dv_res", 1.0, marker="o", ms=3, color="#2c7fb8")
        a_dr.set_title("Doppler resolution", fontsize=10)
        a_dr.set_ylabel("m/s")
        has_tgt = line(a_da, "sig_v_tgt", 1.0, marker="o", ms=3,
                       color="#2c7fb8", label="at target SNR")
        line(a_da, "sig_v_thr", 1.0, marker="s", ms=3, ls="--",
             color="#e67e22", label="at required SNR")
        a_da.set_title("Doppler accuracy (1 sigma)", fontsize=10)
        a_da.set_ylabel("m/s")
        if not has_tgt:
            a_da.text(0.5, 0.5, "needs N >= 2", ha="center", va="center",
                      transform=a_da.transAxes, color="gray")
        for ax in axes:
            ax.set_xlabel(sw["unit_label"], fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            if use_log:
                ax.set_xscale("log")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
        if not ok:
            a_rng.text(0.5, 0.5, "No valid point", ha="center",
                       va="center", transform=a_rng.transAxes)
        fig.tight_layout()
        sw["canvas"].draw_idle()

    sw_ctrl = ttk.Frame(max_tab)
    sw_ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    r1 = ttk.Frame(sw_ctrl)
    r1.pack(fill="x")
    r2 = ttk.Frame(sw_ctrl)
    r2.pack(fill="x", pady=(4, 0))
    ttk.Label(r1, text="Optimise").pack(side="left")
    sw_box = ttk.Combobox(r1, textvariable=sw_param, state="readonly",
                          width=34, values=[p[0] for p in
                                            SWEEP_PARAMS.values()])
    sw_box.pack(side="left", padx=(6, 12))
    ttk.Button(r1, text="Run sweep", command=run_max_range).pack(
        side="left")
    ttk.Label(r1, text="(uses the current inputs on the left; Calculate is "
              "not needed)", foreground="gray45").pack(
        side="left", padx=(10, 0))
    ttk.Label(r2, text="from").pack(side="left")
    ttk.Entry(r2, textvariable=sw_lo, width=9).pack(
        side="left", padx=(4, 8))
    ttk.Label(r2, text="to").pack(side="left")
    ttk.Entry(r2, textvariable=sw_hi, width=9).pack(
        side="left", padx=(4, 4))
    sw_unit = ttk.Label(r2, width=5, foreground="gray45")
    sw_unit.pack(side="left")
    ttk.Label(r2, text="steps").pack(side="left", padx=(8, 0))
    ttk.Entry(r2, textvariable=sw_steps, width=5).pack(
        side="left", padx=(4, 8))
    ttk.Checkbutton(r2, text="log spacing", variable=sw_log).pack(
        side="left")
    sw_match_chk = ttk.Checkbutton(
        r2, text="B_rx follows PW", variable=sw_match)
    sw_match_chk.pack(side="left", padx=(8, 0))

    if HAVE_PLOT:
        sw["fig"] = Figure(figsize=(8, 6.5), dpi=100)
        sw["canvas"] = FigureCanvasTkAgg(sw["fig"], master=max_tab)
        sw["canvas"].get_tk_widget().grid(row=1, column=0, sticky="nsew")
    else:
        ttk.Label(max_tab, text="Install numpy and matplotlib for the "
                  "plot (pip install numpy matplotlib).").grid(
            row=1, column=0, sticky="nw")
    ttk.Label(max_tab, textvariable=sw_summary,
              font=("Segoe UI", 10, "bold")).grid(
        row=2, column=0, sticky="w", pady=(6, 4))
    sw_box.bind("<<ComboboxSelected>>", sw_defaults)
    sw_defaults()

    # ---- Transmitter / signal chain inputs (left column, "Transmitter") ----
    chain_tab.columnconfigure(0, weight=1)
    chain_tab.rowconfigure(1, weight=1)
    cv = {}                               # chain input variables
    chain_rows = {}
    type_names = {v: k for k, v in CHAIN_TYPES.items()}
    lo_names = {v: k for k, v in RX_LO_TYPES.items()}
    lock_names = {v: k for k, v in LOCK_POINTS.items()}
    corr_names = {v: k for k, v in DIG_CORR.items()}
    tx_f.columnconfigure(0, weight=1)
    tx_f.columnconfigure(1, weight=1)
    cv["type"] = tk.StringVar(value=CHAIN_TYPES[CHAIN_DEFAULT["type"]])
    top = ttk.Frame(tx_f)
    top.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
    ttk.Label(top, text="Transmitter type").pack(side="left")
    type_box = ttk.Combobox(top, textvariable=cv["type"], state="readonly",
                            width=34, values=list(CHAIN_TYPES.values()))
    type_box.pack(side="left", padx=(12, 0))
    ttk.Label(tx_f, text="Phase noise of every block adds (independent "
              "processes); results and plots: 'Signal chain' tab.\n"
              "Oscillator L(f): datasheet points 'offset kHz:dBc/Hz, ...' "
              "(flat beyond the last point). Presets are rough.",
              foreground="gray45").grid(row=3, column=0, columnspan=2,
                                        sticky="w", pady=(4, 0))

    def ch_frame(row, col, title, span=1):
        f = ttk.LabelFrame(tx_f, text=title, padding=6)
        f.grid(row=row, column=col, columnspan=span, sticky="nsew",
               padx=(0, 6 if col == 0 and span == 1 else 0), pady=(0, 4))
        return f

    def ch_entry(frame, key, label, default, hint=""):
        r = chain_rows.get(frame, 0)
        ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w")
        cv[key] = tk.StringVar(value=f"{default:g}")
        e = ttk.Entry(frame, textvariable=cv[key], width=7)
        e.grid(row=r, column=1, sticky="w", padx=(6, 0), pady=1)
        if hint:
            ttk.Label(frame, text=hint, foreground="gray45").grid(
                row=r, column=2, sticky="w", padx=(6, 0))
        chain_rows[frame] = r + 1
        return e

    def osc_input(frame, key, presets, default_text, label="L(f)"):
        """Preset selector + editable 'kHz:dBc/Hz' point list."""
        r = chain_rows.get(frame, 0)
        names = list(presets) + [CUSTOM]
        start = next((k for k, v in presets.items() if v == default_text),
                     CUSTOM)
        cv[key + "_preset"] = tk.StringVar(value=start)
        ttk.Label(frame, text="Preset").grid(row=r, column=0, sticky="w")
        box = ttk.Combobox(frame, textvariable=cv[key + "_preset"],
                           state="readonly", width=33, values=names)
        box.grid(row=r, column=1, columnspan=2, sticky="w", padx=(6, 0),
                 pady=1)
        cv[key] = tk.StringVar(value=default_text)
        ttk.Label(frame, text=label).grid(row=r + 1, column=0, sticky="w")
        ent = ttk.Entry(frame, textvariable=cv[key], width=36)
        ent.grid(row=r + 1, column=1, columnspan=2, sticky="w", padx=(6, 0),
                 pady=1)
        chain_rows[frame] = r + 2

        def on_preset(*_):
            name = cv[key + "_preset"].get()
            if name in presets and cv[key].get() != presets[name]:
                cv[key].set(presets[name])

        def on_text(*_):
            name = cv[key + "_preset"].get()
            if presets.get(name) != cv[key].get():
                match = next((k for k, v in presets.items()
                              if v == cv[key].get()), CUSTOM)
                if match != name:
                    cv[key + "_preset"].set(match)

        box.bind("<<ComboboxSelected>>", on_preset)
        cv[key].trace_add("write", on_text)
        return box, ent

    src_f = ch_frame(1, 0, "Source oscillator (exciter)")
    osc_input(src_f, "src_pn", OSC_PRESETS, CHAIN_DEFAULT["src_pn"])

    amp_f = ch_frame(1, 1, "Power amplifier")
    cv["amp"] = tk.StringVar(value=CHAIN_DEFAULT["amp"])
    ttk.Label(amp_f, text="Type").grid(row=0, column=0, sticky="w")
    amp_box = ttk.Combobox(amp_f, textvariable=cv["amp"], state="readonly",
                           width=17, values=list(AMP_TYPES))
    amp_box.grid(row=0, column=1, columnspan=2, sticky="w", padx=(6, 0))
    chain_rows[amp_f] = 1
    amp_entries = [
        ch_entry(amp_f, "amp_push", "Phase pushing (deg/%)",
                 CHAIN_DEFAULT["amp_push"]),
        ch_entry(amp_f, "amp_supply", "Supply jitter (% rms)",
                 CHAIN_DEFAULT["amp_supply"]),
        ch_entry(amp_f, "amp_add", "Additive jitter (deg rms)",
                 CHAIN_DEFAULT["amp_add"])]

    mag_f = ch_frame(1, 1, "Magnetron")
    ch_entry(mag_f, "mag_fjit_khz", "P2P jitter (kHz rms)",
             CHAIN_DEFAULT["mag_fjit_khz"])
    ch_entry(mag_f, "mag_chirp_khz", "Intrapulse (kHz rms)",
             CHAIN_DEFAULT["mag_chirp_khz"])
    ttk.Label(mag_f, text="intrapulse: linear f change\n(end - start), "
              "random/pulse", foreground="gray45").grid(
        row=chain_rows[mag_f], column=0, columnspan=2, sticky="w")
    chain_rows[mag_f] += 1

    # calculator: fractional stability (e.g. 7e-7) -> P2P / intrapulse
    calc_f = ttk.LabelFrame(mag_f, text="From frequency stability",
                            padding=4)
    calc_f.grid(row=chain_rows[mag_f], column=0, columnspan=3, sticky="ew",
                pady=(4, 0))
    chain_rows[mag_f] += 1
    calc_v = {"stab": tk.StringVar(value="7e-7"),
              "kind": tk.StringVar(value="peak"),
              "fr": tk.StringVar(value="100")}
    ttk.Label(calc_f, text="Stability df/f").grid(row=0, column=0,
                                                   sticky="w")
    ttk.Entry(calc_f, textvariable=calc_v["stab"], width=8).grid(
        row=0, column=1, sticky="w", padx=(6, 0), pady=1)
    ttk.Combobox(calc_f, textvariable=calc_v["kind"], state="readonly",
                 width=5, values=["peak", "rms"]).grid(
        row=0, column=2, sticky="w", padx=(4, 0))
    ttk.Label(calc_f, text="Ripple freq. (Hz)").grid(row=1, column=0,
                                                      sticky="w")
    ttk.Entry(calc_f, textvariable=calc_v["fr"], width=8).grid(
        row=1, column=1, sticky="w", padx=(6, 0), pady=1)
    calc_msg = tk.StringVar(value="uses Frequency and PW of the Radar "
                            "section")
    ttk.Label(calc_f, textvariable=calc_msg, foreground="gray35",
              wraplength=230, justify="left").grid(
        row=3, column=0, columnspan=3, sticky="w", pady=(2, 0))

    def run_stab_calc():
        try:
            stab = float(calc_v["stab"].get().replace(",", "."))
            fr = float(calc_v["fr"].get().replace(",", "."))
            f0 = float(vars_["freq"].get().replace(",", ".")) * 1e9
            pw_s = float(vars_["pw"].get().replace(",", ".")) * 1e-6
            if not (stab > 0 and fr >= 0 and f0 > 0 and pw_s > 0):
                raise ValueError
        except (ValueError, KeyError):
            calc_msg.set("Enter stability > 0, ripple >= 0 and valid "
                         "Frequency / PW.")
            return
        p2p, intra, a, frpw = magnetron_jitter_from_stability(
            stab, f0, pw_s, fr, calc_v["kind"].get())
        cv["mag_fjit_khz"].set(f"{p2p:.4g}")
        cv["mag_chirp_khz"].set(f"{intra:.4g}")
        msg = (f"peak dev. A = {a/1e3:.4g} kHz at {f0/1e9:g} GHz\n"
               f"P2P = A/sqrt2 = {p2p:.4g} kHz rms\n"
               f"intrapulse = sqrt2 A |sin(pi fr PW)| = {intra:.4g} kHz rms")
        if frpw > 0.25:
            msg += ("\n! ripple period < 4 PW: the change within the pulse "
                    "is not linear (model approximation)")
        calc_msg.set(msg)

    ttk.Button(calc_f, text="Fill P2P / intrapulse",
               command=run_stab_calc).grid(row=2, column=0, columnspan=3,
                                            sticky="w", pady=(2, 0))

    # COHO selection: two side by side blocks (+ none), inputs below
    coho_f = ch_frame(2, 0, "Receive reference (COHO)", span=2)
    cv["coho_mode"] = tk.StringVar(value=CHAIN_DEFAULT["coho_mode"])
    sel = ttk.Frame(coho_f)
    sel.grid(row=0, column=0, sticky="w", pady=(0, 4))
    coho_btns = []
    for key, text in (("analog", "Analog COHO\nIF osc. locked"),
                      ("digital", "Digital COHO\nTX burst -> ADC"),
                      ("none", "None")):
        b = ttk.Radiobutton(sel, text=text, value=key,
                            variable=cv["coho_mode"], style="Toolbutton")
        b.pack(side="left", padx=(0, 4), ipadx=4, ipady=2, fill="y")
        coho_btns.append(b)

    ana_f = ttk.Frame(coho_f)
    ana_f.grid(row=1, column=0, sticky="nw")
    ttk.Label(ana_f, text="Lock point").grid(row=0, column=0, sticky="w")
    cv["lock_point"] = tk.StringVar(
        value=LOCK_POINTS[CHAIN_DEFAULT["lock_point"]])
    lock_box = ttk.Combobox(ana_f, textvariable=cv["lock_point"],
                            state="readonly", width=12,
                            values=list(LOCK_POINTS.values()))
    lock_box.grid(row=0, column=1, sticky="w", padx=(6, 0))
    chain_rows[ana_f] = 1
    ch_entry(ana_f, "mag_lock", "Lock residual (deg rms)",
             CHAIN_DEFAULT["mag_lock"])
    osc_input(ana_f, "coho_pn", COHO_PRESETS, CHAIN_DEFAULT["coho_pn"],
              label="COHO L(f)")

    dig_f = ttk.Frame(coho_f)
    dig_f.grid(row=1, column=0, sticky="nw")
    ttk.Label(dig_f, text="Correction").grid(row=0, column=0, sticky="w")
    cv["dig_corr"] = tk.StringVar(value=DIG_CORR[CHAIN_DEFAULT["dig_corr"]])
    corr_box = ttk.Combobox(dig_f, textvariable=cv["dig_corr"],
                            state="readonly", width=14,
                            values=list(DIG_CORR.values()))
    corr_box.grid(row=0, column=1, sticky="w", padx=(6, 0))
    chain_rows[dig_f] = 1
    ch_entry(dig_f, "dig_err", "Meas. error (deg rms)",
             CHAIN_DEFAULT["dig_err"])
    cv["dig_memory"] = tk.BooleanVar(value=CHAIN_DEFAULT["dig_memory"])
    ttk.Checkbutton(dig_f, text="Phase memory (2nd trip coherent)",
                    variable=cv["dig_memory"]).grid(
        row=chain_rows[dig_f], column=0, columnspan=2, sticky="w")
    chain_rows[dig_f] += 1

    none_lbl = ttk.Label(coho_f, text="No reference: random start phase "
                         "-> no coherent gain.", foreground="#c0392b")
    none_lbl.grid(row=1, column=0, sticky="nw")

    lo_f = ch_frame(2, 0, "Receiver LO", span=2)
    cv["rx_lo"] = tk.StringVar(value=RX_LO_TYPES[CHAIN_DEFAULT["rx_lo"]])
    ttk.Label(lo_f, text="RX LO").grid(row=0, column=0, sticky="w")
    lo_box = ttk.Combobox(lo_f, textvariable=cv["rx_lo"], state="readonly",
                          width=33, values=list(RX_LO_TYPES.values()))
    lo_box.grid(row=0, column=1, columnspan=2, sticky="w", padx=(6, 0))
    chain_rows[lo_f] = 1
    lo_entries = osc_input(lo_f, "lo_pn", OSC_PRESETS, CHAIN_DEFAULT["lo_pn"],
                           label="RX LO L(f)")

    def chain_spec():
        def num(key, label, lo=None):
            raw = cv[key].get().strip().replace(",", ".")
            try:
                v = float(raw)
            except ValueError:
                raise ValueError(f"Transmitter - {label}: not a valid "
                                 "number.") from None
            if not math.isfinite(v) or (lo is not None and v < lo):
                raise ValueError(f"Transmitter - {label}: must be "
                                 f">= {lo:g}." if lo is not None else
                                 f"Transmitter - {label}: invalid.")
            return v

        def pn(key, label):
            text = cv[key].get().strip()
            OscPN.parse(text, label)           # validate (raises ValueError)
            return text

        mag = type_names[cv["type"].get()] == "magnetron"
        mode = cv["coho_mode"].get()
        indep = lo_names[cv["rx_lo"].get()] == "independent"
        spec = {
            "type": type_names[cv["type"].get()],
            "src_pn": pn("src_pn", "STALO" if mag else "Source"),
            "amp": cv["amp"].get(),
            "amp_push": num("amp_push", "phase pushing", 0),
            "amp_supply": num("amp_supply", "supply jitter", 0),
            "amp_add": num("amp_add", "additive jitter", 0),
            "rx_lo": lo_names[cv["rx_lo"].get()],
            "lo_pn": (pn("lo_pn", "RX LO") if indep and not mag
                      else CHAIN_DEFAULT["lo_pn"]),
            "mag_fjit_khz": num("mag_fjit_khz", "magnetron P2P jitter", 0),
            "mag_chirp_khz": num("mag_chirp_khz",
                                 "magnetron intrapulse change", 0),
            "coho_mode": mode,
            "coho": mode != "none",
            "lock_point": lock_names[cv["lock_point"].get()],
            "mag_lock": num("mag_lock", "lock residual", 0),
            "coho_pn": (pn("coho_pn", "COHO") if mag and mode == "analog"
                        else CHAIN_DEFAULT["coho_pn"]),
            "dig_corr": corr_names[cv["dig_corr"].get()],
            "dig_err": num("dig_err", "digital COHO measurement error", 0),
            "dig_memory": bool(cv["dig_memory"].get()),
        }
        return spec

    def on_chain_layout(*_):
        mag = type_names[cv["type"].get()] == "magnetron"
        src_f.config(text="STALO" if mag else "Source oscillator (exciter)")
        if mag:
            amp_f.grid_remove()
            lo_f.grid_remove()
            mag_f.grid()
            coho_f.grid()
        else:
            mag_f.grid_remove()
            coho_f.grid_remove()
            amp_f.grid()
            lo_f.grid()
        mode = cv["coho_mode"].get()
        for frame, key in ((ana_f, "analog"), (dig_f, "digital"),
                           (none_lbl, "none")):
            if mode == key:
                frame.grid()
            else:
                frame.grid_remove()
        indep = lo_names[cv["rx_lo"].get()] == "independent"
        for e in lo_entries:
            e.state(["!disabled"] if indep else ["disabled"])
        no_amp = cv["amp"].get() == "None (source direct)"
        for e in amp_entries:
            e.state(["disabled"] if no_amp else ["!disabled"])
        schedule()

    def on_amp_type(*_):
        push, supply, add = AMP_TYPES[cv["amp"].get()]
        for key, v in (("amp_push", push), ("amp_supply", supply),
                       ("amp_add", add)):
            cv[key].set(f"{v:g}")
        on_chain_layout()

    type_box.bind("<<ComboboxSelected>>", on_chain_layout)
    lo_box.bind("<<ComboboxSelected>>", on_chain_layout)
    amp_box.bind("<<ComboboxSelected>>", on_amp_type)
    for b in coho_btns:
        b.config(command=on_chain_layout)
    mag_f.grid_remove()
    coho_f.grid_remove()

    chv = {"fig": None, "canvas": None, "res": None}

    def redraw_chain(*_):
        if chv["fig"] is None or chv["res"] is None or not tab_is(chain_tab):
            return
        try:
            draw_chain(chv["fig"], chv["res"])
        except (ValueError, ZeroDivisionError) as exc:
            chv["fig"].clear()
            chv["fig"].text(0.05, 0.9, f"Signal chain view not available: "
                            f"{exc}", va="top", fontsize=10)
        chv["canvas"].draw_idle()

    if HAVE_PLOT:
        chv["fig"] = Figure(figsize=(9, 10), dpi=100)
        chv["canvas"] = FigureCanvasTkAgg(chv["fig"], master=chain_tab)
        chv["canvas"].get_tk_widget().grid(row=1, column=0, sticky="nsew")
        ch_bar = ttk.Frame(chain_tab)
        ch_bar.grid(row=2, column=0, sticky="ew")
        NavigationToolbar2Tk(chv["canvas"], ch_bar)
    else:
        ttk.Label(chain_tab, text="Install numpy and matplotlib for the "
                  "signal chain view.").grid(row=1, column=0, sticky="nw")

    def on_tab_changed(_event=None):
        redraw_mf()
        redraw_geo()
        redraw_chain()
        if tab_is(max_tab) and sw["rows"] is None:
            run_max_range(quiet=True)       # first visit: show a sweep

    notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

    text = tk.Text(out, font=("Consolas", 10), wrap="word",
                   relief="flat", padx=8, pady=8, state="disabled")
    scroll = ttk.Scrollbar(out, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    text.grid(row=0, column=0, sticky="nsew")
    scroll.grid(row=0, column=1, sticky="ns")
    text.tag_configure("ok", foreground="#1a7f37", font=("Consolas", 11, "bold"))
    text.tag_configure("bad", foreground="#c0392b", font=("Consolas", 11, "bold"))
    text.tag_configure("warn", foreground="#b9770e", font=("Consolas", 11, "bold"))

    vars_, labels, entries = {}, {}, {}
    table_flags, table_checks = {}, {}
    rows = {}

    def add_entry(frame, key, label, default, typical=None, table_key=False):
        r = rows.get(frame, 0)
        lbl = ttk.Label(frame, text=label)
        lbl.grid(row=r, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=default)
        ent = ttk.Entry(frame, textvariable=var, width=10)
        ent.grid(row=r, column=1, sticky="ew", pady=2, padx=(12, 0))
        if table_key:
            flag = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(frame, text="table", variable=flag,
                                  command=lambda k=key: on_table_toggle(k))
            chk.grid(row=r, column=2, sticky="w", padx=(8, 0))
            table_flags[key], table_checks[key] = flag, chk
        if typical:
            ttk.Label(frame, text=typical, foreground="gray45").grid(
                row=r, column=3, sticky="w", padx=(8, 0))
        vars_[key], labels[key], entries[key] = var, lbl, ent
        rows[frame] = r + 1
        return r

    def add_combo(frame, label, values, default):
        r = rows.get(frame, 0)
        ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=default)
        box = ttk.Combobox(frame, textvariable=var, values=values,
                           state="readonly", width=10)
        box.grid(row=r, column=1, sticky="ew", pady=2, padx=(12, 0))
        rows[frame] = r + 1
        return var, box

    def add_readonly(frame, key, label):
        """Read-only text row spanning the value and hint columns."""
        r = rows.get(frame, 0)
        lbl = ttk.Label(frame, text=label)
        lbl.grid(row=r, column=0, sticky="w", pady=2)
        var = tk.StringVar(value="")
        ent = ttk.Entry(frame, textvariable=var, width=10)
        ent.grid(row=r, column=1, columnspan=3, sticky="ew", pady=2,
                 padx=(12, 0))
        ent.state(["disabled"])
        vars_[key], labels[key], entries[key] = var, lbl, ent
        rows[frame] = r + 1

    def add_slider(frame, key, row, lo, hi):
        """Slider linked to an entry (two-way sync)."""
        dv = tk.DoubleVar()
        try:
            dv.set(float(vars_[key].get()))
        except ValueError:
            dv.set(lo)
        sc = ttk.Scale(frame, from_=lo, to=hi, variable=dv,
                       orient="horizontal", length=190)
        sc.grid(row=row, column=2, columnspan=2, sticky="ew", padx=(8, 0))

        def from_scale(*_):
            if state["sync"]:
                return
            state["sync"] = True
            vars_[key].set(f"{dv.get():.1f}")
            state["sync"] = False

        def from_entry(*_):
            if state["sync"]:
                return
            try:
                v = float(vars_[key].get().replace(",", "."))
            except ValueError:
                return
            if lo <= v <= hi:
                state["sync"] = True
                dv.set(v)
                state["sync"] = False

        dv.trace_add("write", from_scale)
        vars_[key].trace_add("write", from_entry)

    # ---- radar ----
    add_entry(radar_f, "p_dbm", "Peak power (dBm)", "61")
    pri_mode_var, pri_mode_box = add_combo(radar_f, "PRI mode", PRI_MODES,
                                           "PRI Stable")
    add_entry(radar_f, "pri", "PRI (us)", "250")
    add_entry(radar_f, "prf", "PRF (Hz, auto = 1 / PRI)", "")
    pw_mode_var, pw_mode_box = add_combo(radar_f, "PW defined by",
                                         PW_MODES, "Pulse width")
    add_entry(radar_f, "pw", "Pulse width PW (us)", "10")
    add_entry(radar_f, "duty", "Duty cycle (%)", "")
    mop_var, mop_box = add_combo(radar_f, "Modulation on pulse (MOP)",
                                 MOP_OPTIONS, "None")
    add_entry(radar_f, "mop_param", "MOP parameter (n/a)", "")
    bw_row = add_entry(radar_f, "bw", "Receiver bandwidth (MHz)", "0.1")
    suggested = {"mhz": None}

    def use_suggested():
        if suggested["mhz"] is not None:
            vars_["bw"].set(f"{suggested['mhz']:.6g}")

    ttk.Button(radar_f, text="Use", width=5, command=use_suggested).grid(
        row=bw_row, column=2, padx=(8, 0), sticky="w")
    add_entry(radar_f, "freq", "Frequency (GHz)", "16.5")
    add_entry(radar_f, NF_FIELD[0], NF_FIELD[1], NF_FIELD[2],
              f"typ. {NF_FIELD[3]}", table_key=True)
    add_entry(radar_f, "snr", "Required SNR (dB)", "13")
    add_entry(radar_f, "temp", "System temperature (K)", "290")
    cpi_mode_var, cpi_mode_box = add_combo(radar_f, "CPI defined by",
                                           CPI_MODES, "Pulse count")
    add_entry(radar_f, "cpi", "CPI (ms)", "10")
    add_entry(radar_f, "npulse", "Pulses per CPI (N)", "32")
    add_entry(radar_f, "fft", "FFT points (auto, 2^k >= N)", "")
    add_entry(radar_f, "illum", "Illumination time (ms, auto)", "")
    entries["prf"].state(["disabled"])
    entries["fft"].state(["disabled"])
    entries["illum"].state(["disabled"])
    entries["duty"].state(["disabled"])

    # ---- antenna ----
    add_entry(ant_f, "gain", "Gain toward target (dBi)", "41",
              "manual = constant gain", table_key=True)
    pol_var, pol_box = add_combo(ant_f, "Polarisation", POL_OPTIONS, "H")
    r_az = add_entry(ant_f, "ant_az", "Pointing azimuth (deg)", "30")
    add_slider(ant_f, "ant_az", r_az, 0, 360)
    r_el = add_entry(ant_f, "ant_el", "Pointing elevation (deg)", "2")
    add_slider(ant_f, "ant_el", r_el, -10, 90)

    def point_at_target():
        vars_["ant_az"].set(vars_["tgt_az"].get())
        vars_["ant_el"].set(vars_["tgt_el"].get())

    ttk.Button(ant_f, text="Point antenna at target",
               command=point_at_target).grid(
        row=rows[ant_f], column=0, columnspan=4, sticky="ew", pady=(6, 0))
    rows[ant_f] += 1

    # ---- target ----
    add_entry(tgt_f, "rcs", "RCS (m^2)", "0.1", "from aspect + pol",
              table_key=True)
    r_rng = add_entry(tgt_f, "tgt_range", "Range (km)", "30")
    add_slider(tgt_f, "tgt_range", r_rng, 0.1, 200)
    r_taz = add_entry(tgt_f, "tgt_az", "Azimuth (deg)", "30")
    add_slider(tgt_f, "tgt_az", r_taz, 0, 360)
    r_tel = add_entry(tgt_f, "tgt_el", "Elevation (deg)", "2")
    add_slider(tgt_f, "tgt_el", r_tel, -10, 90)
    add_entry(tgt_f, "tgt_heading", "Heading (deg)", "210",
              "tgt az + 180 = nose-on")
    add_entry(tgt_f, "tgt_speed", "Speed (m/s)", "200", "along heading")
    r0 = rows[tgt_f]
    ttk.Label(tgt_f, text="What the radar finds (read-only)",
              foreground="gray45").grid(row=r0, column=0, columnspan=4,
                                        sticky="w", pady=(8, 0))
    rows[tgt_f] = r0 + 1
    measure_keys = (("m_radial", "radial", "Radial velocity"),
                    ("m_angrate", "angrate", "LOS angular rate"),
                    ("m_dop_true", "dop_true", "Doppler, true"),
                    ("m_range", "range_found", "Range found"),
                    ("m_dop", "dop_found", "Doppler found"),
                    ("m_ru", "ru", "Unambiguous range"),
                    ("m_dopu", "dop_unamb", "Unambiguous Doppler"))
    for key, _t, label in measure_keys:
        add_readonly(tgt_f, key, label)

    # ---- losses ----
    for key, label, default, typical, _scope in LOSS_FIELDS:
        add_entry(loss_f, key, label, default, f"typ. {typical}",
                  table_key=True)
    a_key, a_label, a_default, a_typ = ATMOS_FIELD
    add_entry(loss_f, a_key, a_label, a_default, f"typ. {a_typ}",
              table_key=True)
    for key, label, default, typical in RAIN_FIELDS:
        add_entry(loss_f, key, label, default, f"typ. {typical}")

    loss_keys = [k for k, *_ in LOSS_FIELDS] + [a_key, NF_FIELD[0]]
    table_keys = ["gain", "rcs"] + loss_keys

    # ---- helpers ----
    def parse(key, optional=False):
        raw = vars_[key].get().strip().replace(",", ".")
        name = labels[key].cget("text")
        if raw == "":
            if optional:
                return None
            raise ValueError(f"{name}: value is required.")
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{name}: not a valid number.") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        return value

    def parse_list(key):
        """Comma separated list of positive numbers (dot as decimal mark)."""
        name = labels[key].cget("text")
        raw = vars_[key].get().replace(";", ",").replace(" ", ",")
        parts = [x for x in raw.split(",") if x.strip()]
        if not parts:
            raise ValueError(f"{name}: at least one value is required.")
        values = []
        for part in parts:
            try:
                v = float(part)
            except ValueError:
                raise ValueError(f"{name}: '{part}' is not a valid number "
                                 "(use a dot as decimal mark).") from None
            if not math.isfinite(v):
                raise ValueError(f"{name}: values must be finite numbers.")
            values.append(v)
        return values

    def parse_int(key):
        value = parse(key)
        if value < 1 or abs(value - round(value)) > 1e-9:
            raise ValueError(f"{labels[key].cget('text')}: must be a "
                             "positive integer.")
        return int(round(value))

    def set_quiet(key, value):
        """Set an entry without triggering a recalculation."""
        state["suppress"] = True
        vars_[key].set(value)
        state["suppress"] = False

    def on_table_toggle(key):
        entries[key].state(["disabled"] if table_flags[key].get()
                           else ["!disabled"])
        schedule()

    def apply_table_availability(first_time=False):
        for key in table_keys:
            available = TABLE_NAME.get(key, key) in state["tables"]
            if available:
                table_checks[key].state(["!disabled"])
                if first_time and not state["tables"][
                        TABLE_NAME.get(key, key)].get("dummy"):
                    table_flags[key].set(True)   # DUMMY tables stay opt-in
            else:
                table_flags[key].set(False)
                table_checks[key].state(["disabled"])
            entries[key].state(["disabled"] if table_flags[key].get()
                               else ["!disabled"])

    def reload_tables():
        state["tables"], state["table_msgs"] = load_tables(table_dir)
        apply_table_availability()
        calculate()

    def update_suggested(*_):
        try:
            mop = mop_var.get()
            pw = parse("pw")
            param = None if mop == "None" else parse("mop_param")
            b_sig, _note = signal_bandwidth_hz(pw, mop, param)
            suggested["mhz"] = b_sig / 1e6
            labels["bw"].config(
                text=f"Receiver bandwidth (MHz) "
                     f"(suggested: {suggested['mhz']:.4g})")
        except (ValueError, ZeroDivisionError, OverflowError):
            suggested["mhz"] = None
            labels["bw"].config(text="Receiver bandwidth (MHz) (suggested: -)")

    saved_param = dict(MOP_PARAM_DEFAULT)
    current = {"mop": "None"}

    def on_mop_change(*_):
        old = current["mop"]
        if old != "None":
            saved_param[old] = vars_["mop_param"].get()
        new = mop_var.get()
        current["mop"] = new
        if new == "None":
            labels["mop_param"].config(text="MOP parameter (n/a)")
            set_quiet("mop_param", "")
            entries["mop_param"].state(["disabled"])
        else:
            labels["mop_param"].config(text=MOP_PARAM_LABEL[new])
            entries["mop_param"].state(["!disabled"])
            set_quiet("mop_param", saved_param[new])
        update_suggested()
        schedule()

    def spread(values, fmt):
        a, b = fmt.format(min(values)), fmt.format(max(values))
        return a if a == b else f"{a} - {b}"

    def update_derived(*_):
        """PRF, FFT size and the computed one of CPI time / pulse count."""
        if state["suppress"]:
            return
        switch = pri_mode_var.get() == "PRI Switcher"
        count = cpi_mode_var.get() == "Pulse count"
        shown = "cpi" if count else "npulse"       # the computed entry
        try:
            pris = parse_list("pri") if switch else [parse("pri")]
            if min(pris) <= 0:
                raise ValueError("PRI must be positive.")
            set_quiet("prf", ", ".join(f"{1e6 / p:.2f}" for p in pris))
            if count:
                n = parse_int("npulse")
                ns = [n] * len(pris)
                cpis = [n * p / 1000.0 for p in pris]
                set_quiet("cpi", spread(cpis, "{:.4g}"))
            else:
                cpi = parse("cpi")
                cpis = [cpi] * len(pris)
                ns = [math.floor(cpi * 1000.0 / p + 1e-9) for p in pris]
                set_quiet("npulse", spread(ns, "{:d}") if min(ns) >= 1
                          else "< 1 (CPI shorter than PRI)")
            if min(ns) < 1:
                raise ValueError("no pulse in CPI")
            set_quiet("fft", ", ".join(str(fft_size(n)) for n in ns))
            # Illumination time: one CPI (PRI Stable) or the sum of the
            # CPIs of all PRIs in the switch sequence (PRI Switcher).
            set_quiet("illum", f"{sum(cpis):.4g}")
        except ValueError:
            for key in ("prf", "fft", "illum", shown):
                set_quiet(key, "-")

    def update_pw_duty(*_):
        """PW = DC x PRI (Duty cycle mode) or DC = PW / PRI (PW mode).
        With a PRI list (PRI Switcher) the shortest PRI is used."""
        if state["suppress"]:
            return
        by_duty = pw_mode_var.get() == "Duty cycle"
        shown = "pw" if by_duty else "duty"          # the computed entry
        try:
            switch = pri_mode_var.get() == "PRI Switcher"
            pri = min(parse_list("pri") if switch else [parse("pri")])
            if pri <= 0:
                raise ValueError("PRI must be positive.")
            if by_duty:
                dc = parse("duty")
                if not 0 < dc <= 100:
                    raise ValueError("Duty cycle must be in (0, 100] %.")
                set_quiet("pw", f"{dc / 100.0 * pri:.6g}")
            else:
                set_quiet("duty", f"{parse('pw') / pri * 100.0:.6g}")
        except ValueError:
            set_quiet(shown, "-")

    def on_pw_mode_change(*_):
        by_duty = pw_mode_var.get() == "Duty cycle"
        entries["pw"].state(["disabled"] if by_duty else ["!disabled"])
        entries["duty"].state(["!disabled"] if by_duty else ["disabled"])
        update_pw_duty()
        schedule()

    saved_cpi = {"cpi": "10", "npulse": "32"}
    cpi_cur = {"mode": "CPI time"}

    def on_cpi_mode_change(*_):
        old_key = "cpi" if cpi_cur["mode"] == "CPI time" else "npulse"
        saved_cpi[old_key] = vars_[old_key].get()
        new = cpi_mode_var.get()
        cpi_cur["mode"] = new
        key, other = (("cpi", "npulse") if new == "CPI time"
                      else ("npulse", "cpi"))
        set_quiet(key, saved_cpi[key])
        entries[key].state(["!disabled"])
        entries[other].state(["disabled"])
        update_derived()
        schedule()

    saved_pri = {"PRI Stable": "250", "PRI Switcher": PRI_LIST_DEFAULT}
    pri_cur = {"mode": "PRI Stable"}

    def on_pri_mode_change(*_):
        saved_pri[pri_cur["mode"]] = vars_["pri"].get()
        new = pri_mode_var.get()
        pri_cur["mode"] = new
        if new == "PRI Switcher":
            labels["pri"].config(text="PRI list (us, comma separated)")
            labels["cpi"].config(text="CPI = dwell per PRI (ms)")
            labels["npulse"].config(text="Pulses per dwell (N)")
        else:
            labels["pri"].config(text="PRI (us)")
            labels["cpi"].config(text="CPI (ms)")
            labels["npulse"].config(text="Pulses per CPI (N)")
        set_quiet("pri", saved_pri[new])
        update_derived()
        update_pw_duty()
        schedule()

    def show(msg, status=None):
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", msg)
        if status is not None:
            text.tag_add(status, "1.0", "1.end")
        text.configure(state="disabled")

    def gather():
        mop = mop_var.get()
        switch = pri_mode_var.get() == "PRI Switcher"
        count = cpi_mode_var.get() == "Pulse count"
        return {
            "p_dbm": parse("p_dbm"), "pw": parse("pw"),
            "pri_mode": "switch" if switch else "stable",
            "pri_list": parse_list("pri") if switch else [parse("pri")],
            "mop": mop,
            "mop_param": None if mop == "None" else parse("mop_param"),
            "bw": parse("bw"), "freq": parse("freq"),
            "chain": chain_spec(),
            "snr": parse("snr"), "temp": parse("temp"), "cpi": None if count else parse("cpi"),
            "n_pulses": parse_int("npulse") if count else None,
            "pol": POL_CODE[pol_var.get()],
            "ant_az": parse("ant_az"), "ant_el": parse("ant_el"),
            "gain_table": table_flags["gain"].get(),
            "gain_manual": None if table_flags["gain"].get() else parse("gain"),
            "rcs_table": table_flags["rcs"].get(),
            "rcs_manual": None if table_flags["rcs"].get() else parse("rcs"),
            "tgt_range": parse("tgt_range"), "tgt_az": parse("tgt_az"),
            "tgt_el": parse("tgt_el"), "tgt_heading": parse("tgt_heading"),
            "tgt_speed": parse("tgt_speed"),
            "loss_table": {k: table_flags[k].get() for k in loss_keys},
            "loss_manual": {k: (None if table_flags[k].get() else parse(k))
                            for k in loss_keys},
            "rain_rate": parse("rain_rate"), "rain_path": parse("rain_path"),
            "notes": list(state["table_msgs"]),
        }

    def calculate(live=False):
        state["pending"] = None
        try:
            res = evaluate(gather(), state["tables"])
        except (ValueError, OverflowError, ZeroDivisionError, KeyError) as exc:
            for key, _t, _l in measure_keys:
                set_quiet(key, "-")
            if live:
                show(f"Input error: {exc}\n\n"
                     "Results update automatically when all inputs are valid.")
            else:
                messagebox.showerror("Invalid input", str(exc))
            return
        show(format_result(res, table_dir), res["status_kind"])
        mf["res"] = res
        geo["res"] = res
        chv["res"] = res
        redraw_mf()
        redraw_geo()
        redraw_chain()
        summary = target_summary(res, short=True)
        for key, tkey, _l in measure_keys:
            set_quiet(key, summary[tkey])
        # Show table-derived values in their (disabled) entries
        if table_flags["gain"].get():
            set_quiet("gain", f"{res['gain_dbi']:.2f}")
        if table_flags["rcs"].get():
            set_quiet("rcs", f"{res['rcs_m2']:.4g}")
        for key in loss_keys:
            if table_flags[key].get():
                set_quiet(key, f"{res['loss_values'][key]:.4g}")

    def schedule(*_):
        """Debounced live recalculation."""
        if state["suppress"]:
            return
        if state["pending"] is not None:
            root.after_cancel(state["pending"])
        state["pending"] = root.after(80, lambda: calculate(live=True))

    buttons = ttk.Frame(main)
    buttons.grid(row=1, column=0, sticky="ew", pady=(10, 0), padx=(0, 12))
    buttons.columnconfigure(0, weight=1)
    ttk.Button(buttons, text="Calculate", command=calculate).grid(
        row=0, column=0, sticky="ew")
    ttk.Button(buttons, text="Reload tables", command=reload_tables).grid(
        row=0, column=1, padx=(8, 0))

    # ---- wiring ----
    mop_box.bind("<<ComboboxSelected>>", on_mop_change)
    pri_mode_box.bind("<<ComboboxSelected>>", on_pri_mode_change)
    cpi_mode_box.bind("<<ComboboxSelected>>", on_cpi_mode_change)
    for key in ("pri", "cpi", "npulse"):
        vars_[key].trace_add("write", update_derived)
    for key in ("pri", "pw", "duty"):
        vars_[key].trace_add("write", update_pw_duty)
    pw_mode_box.bind("<<ComboboxSelected>>", on_pw_mode_change)
    pol_box.bind("<<ComboboxSelected>>", schedule)
    vars_["pw"].trace_add("write", update_suggested)
    vars_["mop_param"].trace_add("write", update_suggested)
    for var in vars_.values():
        var.trace_add("write", schedule)
    for var in cv.values():
        var.trace_add("write", schedule)
    root.bind("<Return>", lambda _e: calculate())

    on_mop_change()
    on_chain_layout()
    on_cpi_mode_change()
    on_pw_mode_change()
    apply_table_availability(first_time=True)
    calculate()
    return root


def main():
    build_gui().mainloop()


if __name__ == "__main__":
    main()