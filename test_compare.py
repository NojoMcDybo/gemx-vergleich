"""Self-check for compare.py on synthetic data: python -I test_compare.py"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compare as c

N, LAG, FLIPPED, BOTH = 200, 2, range(50, 60), range(120, 125)  # legs only / legs + arms
BASE = {"pelvis": (0, 0, 1.0), "neck": (0, 0, 1.5), "head": (0, 0, 1.7),
        "r_shoulder": (0.2, 0, 1.45), "r_elbow": (0.25, 0, 1.15), "r_wrist": (0.28, 0, 0.9),
        "l_shoulder": (-0.2, 0, 1.45), "l_elbow": (-0.25, 0, 1.15), "l_wrist": (-0.28, 0, 0.9),
        "r_hip": (0.1, 0, 0.95), "r_knee": (0.1, 0, 0.5), "r_ankle": (0.1, 0, 0.08),
        "l_hip": (-0.1, 0, 0.95), "l_knee": (-0.1, 0, 0.5), "l_ankle": (-0.1, 0, 0.08)}


def xsens_frame(k):
    # every joint moves on its own smooth path, so the pose changes over time
    out = []
    for j, n in enumerate(c.NAMES):
        x, y, z = BASE[n]
        ph = k / 15 + j
        out.append((x + 0.06 * math.sin(ph), y + 0.12 * math.sin(0.7 * ph), z + 0.05 * math.cos(ph)))
    return out


def offset_joints(f):
    """Xsens frame with other joint definitions: right knee 3 cm further down the thigh, right
    hip 2 cm further right."""
    f = f[:]
    hk = c._unit(c._sub(f[c.IDX["r_knee"]], f[c.IDX["r_hip"]]))
    f[c.IDX["r_knee"]] = tuple(p + 0.03 * u for p, u in zip(f[c.IDX["r_knee"]], hk))
    f[c.IDX["r_hip"]] = c._sub(f[c.IDX["r_hip"]], (-0.02, 0, 0))
    return f


def rotation(axis, deg):
    x, y, z = (v / math.sqrt(sum(a * a for a in axis)) for v in axis)
    a = math.radians(deg)
    co, si, t = math.cos(a), math.sin(a), 1 - math.cos(a)
    return [[co + x * x * t, x * y * t - z * si, x * z * t + y * si],
            [y * x * t + z * si, co + y * y * t, y * z * t - x * si],
            [z * x * t - y * si, z * y * t + x * si, co + z * z * t]]


def main():
    rnd = random.Random(1)
    xs = [xsens_frame(k) for k in range(N)]
    T = (1.0, rotation((0.3, -1, 0.5), 140), (0.25, -0.04, 3.0))  # metres, camera frame
    pred = []
    for m in range(N):
        f = xs[max(m - LAG, 0)][:]
        swaps = c.SWAPS["Beine"] * (m in FLIPPED or m in BOTH) + c.SWAPS["Arme"] * (m in BOTH)
        for p, q in swaps:
            f[c.IDX[p]], f[c.IDX[q]] = f[c.IDX[q]], f[c.IDX[p]]
        pred.append([tuple(v + rnd.gauss(0, 0.003) for v in c._apply(T, p)) for p in f])

    # batched numpy fit gives the same per-frame errors as the plain Python one
    fast, slow = c._pa_errs(pred[:20], xs[:20]), [c._pa_err(p, q) for p, q in zip(pred[:20], xs[:20])]
    assert all(abs(x - y) < 1e-9 for x, y in zip(fast, slow)) and len(fast) == 20, (fast, slow)
    summary, series = c.evaluate(xs, pred)
    assert summary["lag_frames"] == LAG, summary["lag_frames"]
    assert abs(summary["scale"] - 1) < 0.02, summary["scale"]
    assert summary["mpjpe_mm"] < 20, summary["mpjpe_mm"]  # rigid fit recovers the pose
    bad = set(FLIPPED) | set(BOTH)
    clean = [e for i, e in enumerate(series["pa_mm"]) if series["k0"] + i + LAG not in bad]
    assert c._mean(clean) < 6, c._mean(clean)  # 3 mm noise per axis
    for name, want in (("Beine", sorted(bad)), ("Arme", list(BOTH))):
        flagged = [series["k0"] + i for i, f in enumerate(series["flips"][name]) if f]
        assert flagged == [m - LAG for m in want], (name, flagged)
    assert summary["angle_rmse_deg"]["Ellbogen R"] < 2, summary["angle_rmse_deg"]
    assert abs(summary["bones"]["Oberarm R"]["ratio"] - 1) < 0.02
    assert summary["pck_pct"]["50"] > 90 and summary["pa_median_mm"] <= summary["pa_p95_mm"]
    assert abs(summary["pose_depth_mm"] / summary["pose_plane_mm"] - 1) < 0.5  # isotropic noise

    # depth-only noise, a linear drift and constant joint offsets (other joint definitions)
    ref = [xsens_frame(k) for k in range(N)]
    shifted = [offset_joints(f) for f in ref]
    pred = []
    for m, f in enumerate(shifted):
        cam = [c._apply(T, p) for p in f]
        pred.append([(x + 0.001 * m + rnd.gauss(0, 0.002), y + rnd.gauss(0, 0.002), z + rnd.gauss(0, 0.02))
                     for x, y, z in cam])
    s2, _ = c.evaluate(ref, pred, max_lag=0)
    assert s2["pose_depth_mm"] > 4 * s2["pose_plane_mm"], (s2["pose_depth_mm"], s2["pose_plane_mm"])
    assert 25 < s2["drift_mm_s"] < 35, s2["drift_mm_s"]  # 1 mm per frame at 30 fps
    # offsets are only defined relative to the rest of the body (shifting every joint alike is a
    # translation the fit absorbs), so check which joints stand out, not their exact size
    top = sorted(s2["offset_mm"], key=s2["offset_mm"].get)[-2:]
    assert set(top) == {"r_knee", "r_hip"}, s2["offset_mm"]
    assert s2["pa_mpjpe_fix_mm"] < s2["pa_mpjpe_mm"]
    assert s2["angle_rmse_fix_deg"]["Knie R"] < s2["angle_rmse_deg"]["Knie R"]
    al = c._pa_align(shifted, ref)  # without noise most of the error goes
    fixed, _ = c.offset_fix(al, ref)
    assert c._mean(c._pa_errs(fixed, ref)) < 0.4 * c._mean(c._pa_errs(al, ref))

    # small helpers
    assert c._pct([3, 1, 2, 4], 0.5) == 2.5 and c._pearson([1, 2, 3], [2, 4, 6.0]) == 1
    assert c._pearson([1, 1], [1, 2]) is None and abs(c._slope([1, 3, 5, 7]) - 2) < 1e-12
    assert abs(c._fit_slope([0, 10, 20], [1, 0, -1]) + 0.1) < 1e-12
    x = [math.sin(k / 10) for k in range(100)]
    g = [0.9 * math.sin((k - 3) / 10) for k in range(100)]  # 3 frames late, 10 % low
    err, lag = c.peak(g, x)
    assert abs(err + 0.1) < 0.01 and lag == 3 / c.FPS, (err, lag)

    # resampling 60 Hz -> 30 fps on a linear motion is exact
    t = [k / 60 - 0.008 for k in range(61)]
    frames = [[(tk, 2 * tk, 0.0)] * len(c.NAMES) for tk in t]
    out = c.resample(t, frames)
    assert len(out) == 30 and all(abs(f[0][0] - k / 30) < 1e-9 for k, f in enumerate(out))
    print("ok")


if __name__ == "__main__":
    main()
