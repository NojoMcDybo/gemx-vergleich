"""Compare GemX 3D joints with Xsens and build a viewer page.

Usage: python compare.py <predimo_dir>   (folders: clapsync.DIRS)
Inputs (from clapsync.py): claps.csv and Xsens_CSV/*_cut.csv in DIRS["sync"], and the
constant-30-fps cut videos GemX was run on. GemX joints go to DIRS["joints"]/<nr>*.csv:
one row per frame of the 30 fps cut video, columns frame;<joint>_x;<joint>_y;<joint>_z with
the joint names in JOINTS, in metres (any coordinate frame; a rigid transform is fitted).
gemx_joints.py writes these files from the GemX hpe_results.pt.
Outputs: DIRS["sync"]/Vergleich.html (data embedded, opens the cut videos by relative path) and
Vergleich.csv (summary per recording). Recordings without GemX data are still shown with the
Xsens skeleton over the video, which doubles as a sync check.
"""
import csv, json, math, os, re, sys
from pathlib import Path

try:
    import numpy as _np  # optional: only speeds up the per-frame fits, results are the same
except ImportError:
    _np = None

from clapsync import DIRS

FPS = 30  # GemX runs on the constant-30-fps cut videos
# joint name -> Xsens segment whose origin is that joint (MVN segment origins sit at the
# proximal joint, e.g. RightLowerLeg starts at the knee)
JOINTS = [("pelvis", "Pelvis"), ("neck", "Neck"), ("head", "Head"),
          ("r_shoulder", "RightUpperArm"), ("r_elbow", "RightForeArm"), ("r_wrist", "RightHand"),
          ("l_shoulder", "LeftUpperArm"), ("l_elbow", "LeftForeArm"), ("l_wrist", "LeftHand"),
          ("r_hip", "RightUpperLeg"), ("r_knee", "RightLowerLeg"), ("r_ankle", "RightFoot"),
          ("l_hip", "LeftUpperLeg"), ("l_knee", "LeftLowerLeg"), ("l_ankle", "LeftFoot")]
NAMES = [n for n, _ in JOINTS]
IDX = {n: i for i, n in enumerate(NAMES)}
BONES = [("pelvis", "neck"), ("neck", "head"), ("neck", "r_shoulder"), ("r_shoulder", "r_elbow"),
         ("r_elbow", "r_wrist"), ("neck", "l_shoulder"), ("l_shoulder", "l_elbow"),
         ("l_elbow", "l_wrist"), ("pelvis", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
         ("pelvis", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle")]
LIMBS = {"Oberarm R": ("r_shoulder", "r_elbow"), "Unterarm R": ("r_elbow", "r_wrist"),
         "Oberarm L": ("l_shoulder", "l_elbow"), "Unterarm L": ("l_elbow", "l_wrist"),
         "Oberschenkel R": ("r_hip", "r_knee"), "Unterschenkel R": ("r_knee", "r_ankle"),
         "Oberschenkel L": ("l_hip", "l_knee"), "Unterschenkel L": ("l_knee", "l_ankle")}
# angle between segment vectors (b - a) and (d - c); 0 deg = segments in line
ANGLES = {"Knie R": ("r_hip", "r_knee", "r_knee", "r_ankle"),
          "Knie L": ("l_hip", "l_knee", "l_knee", "l_ankle"),
          "Hüfte R": ("neck", "pelvis", "r_hip", "r_knee"),
          "Hüfte L": ("neck", "pelvis", "l_hip", "l_knee"),
          "Ellbogen R": ("r_shoulder", "r_elbow", "r_elbow", "r_wrist"),
          "Ellbogen L": ("l_shoulder", "l_elbow", "l_elbow", "l_wrist"),
          "Schulter R": ("neck", "pelvis", "r_shoulder", "r_elbow"),
          "Schulter L": ("neck", "pelvis", "l_shoulder", "l_elbow")}
SWAPS = {"Beine": [("r_hip", "l_hip"), ("r_knee", "l_knee"), ("r_ankle", "l_ankle")],
         "Arme": [("r_shoulder", "l_shoulder"), ("r_elbow", "l_elbow"), ("r_wrist", "l_wrist")]}
FLIP_RATIO, FLIP_MIN = 0.8, 0.02  # swapped side must fit >20 % and >2 cm better
PCK_MM = (50, 100, 150)  # share of joints within these posture errors


def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _dist(a, b): return math.sqrt(_dot(_sub(a, b), _sub(a, b)))
def _mean(xs): return sum(xs) / len(xs) if xs else float("nan")
def _rot(R, p): return tuple(R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2] for i in range(3))


def _top_eigvec(N):
    """Eigenvector of the largest eigenvalue of a symmetric 4x4 matrix (cyclic Jacobi)."""
    a = [row[:] for row in N]
    v = [[float(i == j) for j in range(4)] for i in range(4)]
    for _ in range(50):
        if sum(a[i][j] ** 2 for i in range(4) for j in range(4) if i != j) < 1e-20:
            break
        for p in range(3):
            for q in range(p + 1, 4):
                if abs(a[p][q]) < 1e-30:
                    continue
                th = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = math.copysign(1, th) / (abs(th) + math.sqrt(th * th + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for m in (a, v):  # columns: M J
                    for k in range(4):
                        x, y = m[k][p], m[k][q]
                        m[k][p], m[k][q] = c * x - s * y, s * x + c * y
                for k in range(4):  # rows: J^T A
                    x, y = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * x - s * y, s * x + c * y
    i = max(range(4), key=lambda i: a[i][i])
    return [v[k][i] for k in range(4)]


def similarity(A, B, scale=True):
    """(s, R, t) minimising sum |s R a + t - b|^2 over point pairs (Horn's quaternion method);
    scale=False keeps s = 1 (rigid fit)."""
    n = len(A)
    ca = [sum(p[i] for p in A) / n for i in range(3)]
    cb = [sum(p[i] for p in B) / n for i in range(3)]
    A0 = [_sub(p, ca) for p in A]
    B0 = [_sub(p, cb) for p in B]
    S = [[sum(a[i] * b[j] for a, b in zip(A0, B0)) for j in range(3)] for i in range(3)]
    (xx, xy, xz), (yx, yy, yz), (zx, zy, zz) = S
    w, x, y, z = _top_eigvec([[xx + yy + zz, yz - zy, zx - xz, xy - yx],
                              [yz - zy, xx - yy - zz, xy + yx, zx + xz],
                              [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
                              [xy - yx, zx + xz, yz + zy, -xx - yy + zz]])
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    s = sum(_dot(_rot(R, a), b) for a, b in zip(A0, B0)) / (sum(_dot(a, a) for a in A0) or 1) if scale else 1.0
    rc = _rot(R, ca)
    return s, R, tuple(cb[i] - s * rc[i] for i in range(3))


def _apply(T, p):
    s, R, t = T
    r = _rot(R, p)
    return (s * r[0] + t[0], s * r[1] + t[1], s * r[2] + t[2])


def _frame_err(P, Q): return _mean([_dist(p, q) for p, q in zip(P, Q)])


def _pa_err(P, Q): return _frame_err([_apply(similarity(P, Q), p) for p in P], Q)


def _pa_errs(Ps, Qs):
    """[_pa_err(P, Q) for P, Q in zip(Ps, Qs)]: per-frame mean of _pa_align's joint distances."""
    return [_frame_err(P, Q) for P, Q in zip(_pa_align(Ps, Qs), Qs)]


def _pa_align(Ps, Qs):
    """Each frame P fitted onto Q with its own similarity, all frames at once with numpy when it
    is there (same Horn fit: largest eigenvector of the 4x4 matrix, scale from the rotated points)."""
    if _np is None or not Ps:
        return [[_apply(similarity(P, Q), p) for p in P] for P, Q in zip(Ps, Qs)]
    A, B = _np.asarray(Ps, float), _np.asarray(Qs, float)  # (frames, joints, 3)
    ca, cb = A.mean(1, keepdims=True), B.mean(1, keepdims=True)
    A0, B0 = A - ca, B - cb
    S = _np.einsum("fji,fjk->fik", A0, B0)
    (xx, xy, xz), (yx, yy, yz), (zx, zy, zz) = [[S[:, i, j] for j in range(3)] for i in range(3)]
    N = _np.stack([_np.stack(r, -1) for r in (
        [xx + yy + zz, yz - zy, zx - xz, xy - yx],
        [yz - zy, xx - yy - zz, xy + yx, zx + xz],
        [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
        [xy - yx, zx + xz, yz + zy, -xx - yy + zz])], 1)
    w, x, y, z = _np.moveaxis(_np.linalg.eigh(N)[1][:, :, -1], 1, 0)
    R = _np.stack([_np.stack(r, -1) for r in (
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])], 1)
    RA = _np.einsum("fik,fjk->fji", R, A0)
    den = (A0 * A0).sum((1, 2))
    sc = (RA * B0).sum((1, 2)) / _np.where(den == 0, 1, den)
    return [[tuple(p) for p in f] for f in (sc[:, None, None] * RA + cb).tolist()]


def _sd(xs):
    m = _mean(xs)
    return math.sqrt(_mean([(x - m) ** 2 for x in xs]))


def _pearson(a, b):
    ma, mb = _mean(a), _mean(b)
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else None  # None: flat curve


def _pct(xs, q):
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo = int(i)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (i - lo)


def _fit_slope(xs, ys):
    """Least-squares slope of ys over xs."""
    mx, my = _mean(xs), _mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def _slope(ys): return _fit_slope(range(len(ys)), ys)  # over the frame index


def _unit(v):
    n = math.sqrt(_dot(v, v)) or 1
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def peak(g, x, win=FPS):
    """Highest Xsens value vs the highest GemX value within +-win frames of it: (value error,
    time error in s). Searching only near the Xsens peak keeps repeated movements apart."""
    kx = max(range(len(x)), key=x.__getitem__)
    kg = max(range(max(0, kx - win), min(len(g), kx + win + 1)), key=g.__getitem__)
    return g[kg] - x[kx], (kg - kx) / FPS


# Joints whose position is fixed in the trunk (offset removed in a pelvis-neck-hip frame) and
# limb joints with the joint they hang from (offset removed only along that bone: where the
# joint centre sits on the bone is the part the two body models define differently)
TRUNK = ("pelvis", "neck", "head", "r_shoulder", "l_shoulder", "r_hip", "l_hip")
PARENT = {"r_elbow": "r_shoulder", "r_wrist": "r_elbow", "l_elbow": "l_shoulder", "l_wrist": "l_elbow",
          "r_knee": "r_hip", "r_ankle": "r_knee", "l_knee": "l_hip", "l_ankle": "l_knee"}


def offset_fix(pred, ref, rounds=6):
    """Removes each joint's constant offset GemX - Xsens (different joint definitions of SOMA and
    MVN), measured in the Xsens body. pred is already fitted onto ref frame by frame. The per-frame
    fit spreads one joint's offset over all joints, so estimate, correct, refit, repeat. Offsets
    are relative to the rest of the body: the same shift on every joint is a move the fit absorbs.
    Returns (corrected and refitted frames, offset in metres per joint)."""
    axes = []  # per frame: Xsens trunk axes (right, forward, up) for trunk joints, bone axis for limbs
    for f in ref:
        x = _unit(_sub(f[IDX["r_hip"]], f[IDX["l_hip"]]))
        up = _sub(f[IDX["neck"]], f[IDX["pelvis"]])
        z = _unit(_sub(up, tuple(_dot(up, x) * c for c in x)))
        axes.append([(x, _cross(z, x), z) if n in TRUNK else (_unit(_sub(f[IDX[n]], f[IDX[PARENT[n]]])),)
                     for n in NAMES])
    pred = [list(f) for f in pred]
    total = [[0.0] * len(axes[0][j]) for j in range(len(NAMES))]
    for _ in range(rounds):
        for j in range(len(NAMES)):
            b = [_mean([_dot(_sub(p[j], r[j]), ax[j][i]) for p, r, ax in zip(pred, ref, axes)])
                 for i in range(len(total[j]))]
            total[j] = [t + v for t, v in zip(total[j], b)]
            for f, ax in zip(pred, axes):
                f[j] = _sub(f[j], tuple(sum(v * a[c] for v, a in zip(b, ax[j])) for c in range(3)))
        pred = _pa_align(pred, ref)
    return pred, {n: math.sqrt(sum(v * v for v in total[j])) for j, n in enumerate(NAMES)}


def angle(f, a, b, c, d):
    u, v = _sub(f[IDX[b]], f[IDX[a]]), _sub(f[IDX[d]], f[IDX[c]])
    cos = _dot(u, v) / (math.sqrt(_dot(u, u) * _dot(v, v)) or 1)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def read_xsens(path):
    """Xsens cut CSV -> (t_cut_s list, frames of joint positions in NAMES order)."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    col = {h: i for i, h in enumerate(rows[0])}
    ix = [(col[f"{seg}_x"], col[f"{seg}_y"], col[f"{seg}_z"]) for _, seg in JOINTS]
    t = [float(r[col["t_cut_s"]]) for r in rows[1:]]
    return t, [[tuple(float(r[i]) for i in c) for c in ix] for r in rows[1:]]


def read_joints(path):
    """GemX joints CSV (frame;<joint>_x;_y;_z ...) -> frames in NAMES order."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    col = {h: i for i, h in enumerate(rows[0])}
    missing = [n for n in NAMES if f"{n}_x" not in col]
    if missing:
        raise ValueError(f"{path.name}: missing joints {missing}")
    ix = [(col[f"{n}_x"], col[f"{n}_y"], col[f"{n}_z"]) for n in NAMES]
    return [[tuple(float(r[i]) for i in c) for c in ix] for r in rows[1:] if r]


def resample(t, frames):
    """Linear interpolation of the 60 Hz Xsens frames at the 30 fps video frame times k/FPS."""
    out, j = [], 0
    for k in range(int(t[-1] * FPS + 1e-6) + 1):
        tk = k / FPS
        while j + 1 < len(t) - 1 and t[j + 1] <= tk:
            j += 1
        a = min(max((tk - t[j]) / (t[j + 1] - t[j]), 0.0), 1.0)
        out.append([tuple(p[i] * (1 - a) + q[i] * a for i in range(3))
                    for p, q in zip(frames[j], frames[j + 1])])
    return out


def evaluate(xs, pred, max_lag=3):
    """xs, pred: frames (one point per joint in NAMES) at FPS, both in metres. pred[k + lag] is
    compared with xs[k]; the lag with the lowest error after one rigid fit over all frames is
    kept. The whole-recording fit has no scale: GemX claims metric output, and a free scale
    would shrink it to hide depth errors. PA-MPJPE (per frame) does fit scale."""
    best = None
    for lag in range(-max_lag, max_lag + 1):
        ks = [k for k in range(len(xs)) if 0 <= k + lag < len(pred)]
        A = [p for k in ks for p in pred[k + lag]]
        B = [p for k in ks for p in xs[k]]
        T = similarity(A, B, scale=False)
        err = _mean([_dist(_apply(T, a), b) for a, b in zip(A, B)])
        if best is None or err < best[0]:
            best = (err, lag, T, ks, A, B)
    _, lag, T, ks, A, B = best
    size = similarity(A, B)[0]  # info only: > 1 means GemX moves/is drawn too small
    ref = [xs[k] for k in ks]
    raw = [pred[k + lag] for k in ks]
    al = [[_apply(T, p) for p in f] for f in raw]  # GemX in Xsens space

    glob = [_frame_err(a, r) for a, r in zip(al, ref)]
    pa_al = _pa_align(al, ref)
    pa_j = [[_dist(p, q) for p, q in zip(a, r)] for a, r in zip(pa_al, ref)]
    pa = [_mean(e) for e in pa_j]
    # Try every combination of side swaps: a leg swap alone can look like a body turned by
    # 180 deg, which an arm swap then "fixes" too, so single swaps are not tested one by one.
    combos = [()] + [(n,) for n in SWAPS] + [tuple(SWAPS)]
    flips = {n: [] for n in SWAPS}

    def swapped(a, combo):
        sw = a[:]
        for name in combo:
            for p, q in SWAPS[name]:
                sw[IDX[p]], sw[IDX[q]] = sw[IDX[q]], sw[IDX[p]]
        return sw
    e_sw = {combo: _pa_errs([swapped(a, combo) for a in al], ref) for combo in combos[1:]}
    for i, e in enumerate(pa):
        best_e, best_c = e, ()
        for combo in combos[1:]:
            if e_sw[combo][i] < best_e:
                best_e, best_c = e_sw[combo][i], combo
        hit = best_e < FLIP_RATIO * e and e - best_e > FLIP_MIN
        for n in SWAPS:
            flips[n].append(hit and n in best_c)
    ang_x = {n: [angle(f, *d) for f in ref] for n, d in ANGLES.items()}
    ang_g = {n: [angle(f, *d) for f in raw] for n, d in ANGLES.items()}

    def acc(frames, j):
        return [tuple((frames[k + 1][j][i] - 2 * frames[k][j][i] + frames[k - 1][j][i]) * FPS * FPS
                      for i in range(3)) for k in range(1, len(frames) - 1)]

    acc_x = [acc(ref, j) for j in range(len(NAMES))]
    acc_g = [acc(al, j) for j in range(len(NAMES))]
    bones = {}
    for n, (p, q) in LIMBS.items():
        lx = [_dist(f[IDX[p]], f[IDX[q]]) for f in ref]
        lg = [_dist(f[IDX[p]], f[IDX[q]]) for f in al]
        mg = _mean(lg)
        bones[n] = {"ratio": mg / _mean(lx),
                    "cv": math.sqrt(_mean([(x - mg) ** 2 for x in lg])) / mg}

    # Error along the camera's line of sight vs across it, as RMS per axis (the image plane has
    # two axes, depth one, so equal per-axis values mean no depth weakness). "path" is the pelvis
    # (where the person is), "pose" every other joint relative to the pelvis (posture).
    view = _unit(_rot(T[1], (0.0, 0.0, 1.0)))
    pel = IDX["pelvis"]
    e_path = [_sub(a[pel], r[pel]) for a, r in zip(al, ref)]
    e_pose = [_sub(_sub(a[j], r[j]), ep) for a, r, ep in zip(al, ref, e_path)
              for j in range(len(NAMES)) if j != pel]

    def split(es):
        d2 = [_dot(e, view) ** 2 for e in es]
        return 1000 * math.sqrt(_mean(d2)), 1000 * math.sqrt(_mean([_dot(e, e) - d for e, d in zip(es, d2)]) / 2)

    path_depth, path_plane = split(e_path)
    pose_depth, pose_plane = split(e_pose)
    # drift: linear trend of the pelvis error over the recording, mm per second
    drift = [1000 * FPS * _slope([e[i] for e in e_path]) for i in range(3)]
    fixed, offsets = offset_fix(pa_al, ref)
    pa_fix = _pa_errs(fixed, ref)
    ang_f = {n: [angle(f, *d) for f in fixed] for n, d in ANGLES.items()}
    pa_all = [1000 * e for f in pa_j for e in f]
    peaks = {n: peak(ang_g[n], ang_x[n]) for n in ANGLES}
    summary = {
        "frames": len(ks), "lag_frames": lag, "scale": size,
        "mpjpe_mm": 1000 * _mean(glob), "pa_mpjpe_mm": 1000 * _mean(pa),
        "joint_mm": {n: 1000 * _mean([_dist(a[j], r[j]) for a, r in zip(al, ref)])
                     for j, n in enumerate(NAMES)},
        "flip_pct": {n: 100 * sum(f) / len(f) for n, f in flips.items()},
        "angle_rmse_deg": {n: math.sqrt(_mean([(g - x) ** 2 for g, x in zip(ang_g[n], ang_x[n])]))
                           for n in ANGLES},
        "angle_bias_deg": {n: _mean([g - x for g, x in zip(ang_g[n], ang_x[n])]) for n in ANGLES},
        "bones": bones,
        "acc_xs": _mean([math.sqrt(_dot(a, a)) for j in acc_x for a in j]),
        "acc_gemx": _mean([math.sqrt(_dot(a, a)) for j in acc_g for a in j]),
        "acc_err": _mean([_dist(a, b) for jx, jg in zip(acc_x, acc_g) for a, b in zip(jx, jg)]),
        # distribution of the per-joint posture error (mean hides outliers)
        "pa_median_mm": _pct(pa_all, 0.5), "pa_p95_mm": _pct(pa_all, 0.95),
        "pck_pct": {str(t): 100 * sum(e <= t for e in pa_all) / len(pa_all) for t in PCK_MM},
        # Bland-Altman: limits of agreement = bias +- 1.96 sd
        "angle_sd_deg": {n: _sd([g - x for g, x in zip(ang_g[n], ang_x[n])]) for n in ANGLES},
        "angle_r": {n: _pearson(ang_g[n], ang_x[n]) for n in ANGLES},
        # Bland-Altman trend: change of GemX - Xsens per 10 deg of the mean angle (proportional error)
        "angle_trend_deg10": {n: 10 * _fit_slope([(g + x) / 2 for g, x in zip(ang_g[n], ang_x[n])],
                                                 [g - x for g, x in zip(ang_g[n], ang_x[n])]) for n in ANGLES},
        "angle_rom_err_deg": {n: (max(ang_g[n]) - min(ang_g[n])) - (max(ang_x[n]) - min(ang_x[n])) for n in ANGLES},
        "angle_peak_err_deg": {n: p[0] for n, p in peaks.items()},
        "angle_peak_lag_s": {n: p[1] for n, p in peaks.items()},
        "path_depth_mm": path_depth, "path_plane_mm": path_plane,
        "pose_depth_mm": pose_depth, "pose_plane_mm": pose_plane,
        "drift_mm_s": math.sqrt(_dot(drift, drift)),
        "path_mm": 1000 * _mean([math.sqrt(_dot(e, e)) for e in e_path]),
        # after removing each joint's constant offset (joint definitions of SOMA vs MVN)
        "offset_mm": {n: 1000 * v for n, v in offsets.items()},
        "pa_mpjpe_fix_mm": 1000 * _mean(pa_fix),
        "angle_rmse_fix_deg": {n: math.sqrt(_mean([(g - x) ** 2 for g, x in zip(ang_f[n], ang_x[n])]))
                               for n in ANGLES},
    }
    # GemX joints are in camera space (camera at 0, looking along +z): where the video camera
    # stood in Xsens space, so the viewer can show the skeleton from the video's angle
    cam = {"pos": list(T[2]), "dir": list(_rot(T[1], (0.0, 0.0, 1.0)))}
    series = {"k0": ks[0], "gemx": al, "err_mm": [1000 * e for e in glob],
              "pa_mm": [1000 * e for e in pa], "flips": flips, "angles_gemx": ang_g, "cam": cam}
    return summary, series


def _r(x, nd=1):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    return x


def _flat(frames): return [[round(c, 3) for p in f for c in p] for f in frames]


def meta():
    """Viewer constants shared by Vergleich.html and vergleich_app.py."""
    return {"fps": FPS, "joints": NAMES, "bones": BONES, "angles": list(ANGLES), "swaps": list(SWAPS),
            "angle_at": {n: d[2] for n, d in ANGLES.items()}}  # joint where the angle sits


VIEWS = ("Vorne", "Hinten", "Seite")


def display_name(stem):
    """'028_20260825_BlockB_W_X_GTV_GoPro_Vorne_1080p30fps_BoxhintenAusfallschrittL_cut_30fps'
    -> ('Boxhinten Ausfallschritt L', 'Vorne', '25.08.2026'): name, camera view, date."""
    stem = re.sub(r"_cut(_30fps)?$", "", stem)
    m = re.match(r"\d{3}_(\d{4})(\d\d)(\d\d)_", stem)
    toks = stem.split("_")[2:] if m else stem.split("_")
    toks = [t for t in toks if not re.fullmatch(r"Block\w|[WGXM]|GTV?|FM|GoPro|\d+p\d+fps", t)]
    view = next((t for t in toks if t in VIEWS), "")
    name = " ".join(re.sub(r"(?<=[a-zäöü])(?=[A-ZÄÖÜ])", " ", t) for t in toks if t != view)
    return name or view or stem, view, f"{m[3]}.{m[2]}.{m[1]}" if m else ""


def record(nr, name, video, xs, gem=None, gem_name=None):
    """Viewer entry for one recording; xs/gem = frames at FPS (gem None = Xsens only).
    name is the file stem; the entry gets a readable name, the camera view and the date."""
    title, view, date = display_name(name)
    rec = {"nr": nr, "name": title, "view": view, "date": date, "stem": name, "video": video, "xs": _flat(xs),
           "angles_xs": {n: _r([angle(f, *d) for f in xs]) for n, d in ANGLES.items()},
           "gemx_file": gem_name}
    if gem is not None:
        summary, series = evaluate(xs, gem)
        series["gemx"] = _flat(series["gemx"])
        rec.update(summary=_r(summary, 3), series=dict(_r(series), cam=_r(series["cam"], 3)))
    return rec


def table_row(rec, aufnahme):
    """One line of Vergleich.csv: numbers of the summary, groups flattened."""
    line = {"nr": rec["nr"], "aufnahme": aufnahme, "gemx": rec.get("gemx_file") or ""}
    summary = rec.get("summary")
    if summary:
        line.update({k: v for k, v in summary.items() if not isinstance(v, dict)})
        for group in ("flip_pct", "angle_rmse_deg", "angle_bias_deg", "angle_sd_deg", "angle_r", "angle_trend_deg10",
                      "angle_rom_err_deg", "angle_peak_err_deg", "angle_peak_lag_s",
                      "angle_rmse_fix_deg", "pck_pct", "offset_mm"):
            line.update({f"{group}_{k}": _r(v, 2) for k, v in summary[group].items()})
    return line


def write_table(fh, table):
    fields = list(dict.fromkeys(k for t in table for k in t))
    w = csv.DictWriter(fh, fieldnames=fields, restval="", delimiter=";")
    w.writeheader()
    w.writerows(table)


def page(data):
    """compare_view.html as one self-contained file: nojo design files and data inlined."""
    here = Path(__file__).parent
    read = lambda n: (here / n).read_text(encoding="utf-8")
    return (read("compare_view.html").replace("/*NOJO_CSS*/", read("nojo-ui.css"))
            .replace("/*NOJO_JS*/", read("nojo-ui.js"))
            .replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))


def main():
    root = Path(sys.argv[1])
    sync, gem_dir, cfr = root / DIRS["sync"], root / DIRS["joints"], root / DIRS["cfr"]
    with open(sync / "claps.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter=";") if r.get("mvnx_file")]
    recs, table = [], []
    for r in rows:
        nr, stem = r["file"][:3], r["file"].split(".mp4")[0]
        xs = resample(*read_xsens(sync / "Xsens_CSV" / f"{Path(r['mvnx_file']).stem}_cut.csv"))
        gem = next(iter(sorted(gem_dir.glob(nr + "*.csv"))), None) if gem_dir.exists() else None
        video = Path(os.path.relpath(cfr / f"{stem}_cut_30fps.mp4", sync)).as_posix()
        rec = record(nr, stem, video, xs,
                     read_joints(gem) if gem else None, gem.name if gem else None)
        if gem:
            summary = rec["summary"]
            print(f"{nr}  MPJPE {summary['mpjpe_mm']:.0f} mm, PA {summary['pa_mpjpe_mm']:.0f} mm, "
                  f"lag {summary['lag_frames']}")
        else:
            print(f"{nr}  no GemX joints in {gem_dir.name}/ - Xsens only")
        recs.append(rec)
        table.append(table_row(rec, stem))
    (sync / "Vergleich.html").write_text(page(dict(meta(), recordings=recs)), encoding="utf-8")
    with open(sync / "Vergleich.csv", "w", newline="", encoding="utf-8") as fh:
        write_table(fh, table)


if __name__ == "__main__":
    main()
