"""Cut GoPro videos and Xsens MVNX recordings to the same start/end clap.

The video's audio track is only used to find the claps roughly: in these recordings the audio
drifts against the picture by up to +-250 ms. The video<->Xsens offset comes from motion
instead (video frame differences vs Xsens body speed), refined around each clap, and the
clap itself is taken from the Xsens hand data (moment the hands meet).

Usage:
  python clapsync.py <predimo_dir>            # report only -> claps.csv in DIRS["sync"]
  python clapsync.py <predimo_dir> --cut      # also writes cut videos, frame strips, Xsens CSVs
  python clapsync.py <predimo_dir> --cfr30    # also writes cut videos at constant 30 fps
Folders below <predimo_dir> are in DIRS (shared with compare.py and gemx_joints.py). MVNX
files are matched to videos by their 3-digit number.
"""
import array
import csv
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SR = 48000
WIN = 48  # 1 ms envelope windows
NS = "{http://www.xsens.com/mvn/mvnx}"
# Per-frame MVNX fields written to the cut CSV: field -> (label list key, component names)
EXPORT = {"position": ("segments", "xyz"), "orientation": ("segments", ("q0", "q1", "q2", "q3")),
          "jointAngle": ("joints", ("ang_x", "ang_y", "ang_z"))}
HAND_SEGS = ("RightHand", "LeftHand", "RightForeArm", "LeftForeArm")
GRID_HZ = 100
# Folder layout below the Predimo root (sorted 2026-11-21)
DIRS = {"video": "Videos/BlockB_Original", "mvnx": "Xsens_reprocessed",
        "cut": "Videos/BlockB_geschnitten", "cfr": "Videos/BlockB_geschnitten_30fps",
        "gemx": "GemX/BlockB_Ergebnisse", "joints": "GemX/BlockB_Gelenke_CSV",
        "base": "Sonstiges/Sync_BlockB_018-032",  # clap_overrides.csv
        "sync": "Sonstiges/Sync_BlockB_018-032/_sync"}  # claps.csv, Xsens_CSV, Kontrollbilder, Vergleich.*


def audio_envelope(video):
    # aresample fills the recordings' audio gaps with silence so sample time ~ video pts;
    # highpass keeps the sharp clap transient and drops footsteps/speech fundamentals.
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-ac", "1",
         "-af", "aresample=async=1:first_pts=0,highpass=f=1500", "-ar", str(SR),
         "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    s = array.array("h", raw)
    return [max(abs(x) for x in s[i:i + WIN]) for i in range(0, len(s) - WIN, WIN)]


def onsets(env, min_gap_ms=300):
    """Peaks with a sharp rise: peak / median of the 100 ms before it."""
    cands = []
    for i in range(100, len(env) - 1):
        if env[i] < env[i - 1] or env[i] < env[i + 1]:
            continue
        before = sorted(env[i - 100:i - 5])
        base = before[len(before) // 2] or 1
        cands.append((env[i] / base, env[i], i))
    cands.sort(reverse=True)
    picked = []
    for c in cands:
        if all(abs(c[2] - p[2]) > min_gap_ms for p in picked):
            picked.append(c)
    return picked  # (sharpness, amplitude, ms), strongest first


def frame_times(video):
    # Videos are variable frame rate (dropped frames), so frame index must come from pts.
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=pts_time", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True, check=True).stdout.split()
    return [float(x.rstrip(",")) for x in out]


def nearest_frame(pts, t):
    i = min(range(len(pts)), key=lambda k: abs(pts[k] - t))
    return i, pts[i]


def score(c, peak_amp):
    # Sharp AND loud: a clap is both; a bump or speech is rarely both.
    return min(c[0], 50) / 50 * (c[1] / peak_amp)


def find_claps(video):
    env = audio_envelope(video)
    dur_ms = len(env)
    cands = onsets(env)
    peak_amp = max(c[1] for c in cands)
    ranked = sorted(cands, key=lambda c: -score(c, peak_amp))
    first = [c for c in ranked if c[2] < dur_ms / 2]
    last = [c for c in ranked if c[2] >= dur_ms / 2]
    out = {}
    for name, lst in (("start", first), ("end", last)):
        if not lst:
            out[name] = (None, 0.0)
            continue
        best = lst[0]
        runner = score(lst[1], peak_amp) if len(lst) > 1 else 0.0
        # confidence: how clearly the best candidate beats the next one in its half
        out[name] = (best[2] / 1000, round(score(best, peak_amp) / max(runner, 1e-6), 2))
    return out, ranked[:6], peak_amp


def frame_strip(video, pts, i, png):
    """5 frames around frame i (i-2..i+2) side by side, for visual checking."""
    start = pts[max(i - 2, 0)]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.4f}", "-i", str(video),
                    "-frames:v", "1", "-vf", "scale=384:-1,tile=5x1",
                    "-update", "1", str(png)], check=True)


def cut(video, pts, f0, f1, out, fps=None):
    """Frames f0..f1 inclusive, re-encoded so both clap frames are in the video.
    fps: constant frame rate (missing frames repeat the previous one, audio gaps -> silence)."""
    # Trim by frame index: time-based -ss/-to dropped the end frame (and, after rounding,
    # sometimes the start frame).
    t1 = pts[f1 + 1] if f1 + 1 < len(pts) else pts[f1] + 1 / 30
    vf = f"trim=start_frame={f0}:end_frame={f1 + 1},setpts=PTS-STARTPTS"
    af = f"atrim=start={pts[f0]:.6f}:end={t1:.6f},asetpts=PTS-STARTPTS"
    if fps:
        vf += f",fps={fps}:eof_action=pass"  # default drops the last (clap) frame at EOF
        af = "aresample=async=1:first_pts=0," + af
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", vf, "-af", af,
                    "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-c:a", "aac", str(out)],
                   check=True)


def read_mvnx(path):
    m = {"rate": None, "segments": [], "sensors": [], "joints": [], "frames": []}
    lists = {"segment": "segments", "sensor": "sensors", "joint": "joints"}
    for ev, el in ET.iterparse(path, events=("start", "end")):
        tag = el.tag.replace(NS, "")
        if ev == "start":
            if tag == "subject":
                m["rate"] = float(el.get("frameRate"))
            continue
        if tag in lists and el.get("label"):
            m[lists[tag]].append(el.get("label"))
        elif tag == "frame":
            # identity/tpose frames come first and are calibration poses, not motion
            if el.get("type") == "normal":
                f = {"t": int(el.get("time")) / 1000, "index": el.get("index")}
                for c in el:
                    name = c.tag.replace(NS, "")
                    if name in EXPORT and c.text:
                        f[name] = [float(x) for x in c.text.split()]
                m["frames"].append(f)
            el.clear()
    return m


def _vec(f, field, i):
    return f[field][3 * i:3 * i + 3]


def hand_distance(m):
    r, l = m["segments"].index("RightHand"), m["segments"].index("LeftHand")
    return [math.dist(_vec(f, "position", r), _vec(f, "position", l)) for f in m["frames"]]


def speed_series(m, labels=None):
    """Summed segment speed (m/s) per Xsens frame, all segments or the given ones."""
    F = m["frames"]
    ids = [m["segments"].index(s) for s in labels] if labels else range(len(m["segments"]))
    t = [f["t"] for f in F]
    sp = [0.0] + [sum(math.dist(_vec(b, "position", i), _vec(a, "position", i)) for i in ids)
                  / (tb - ta) for a, b, ta, tb in zip(F, F[1:], t, t[1:])]
    return t, sp


def motion_series(video, pts, upper=False):
    """Mean absolute grey-level change per second at each video frame (picture motion)."""
    w, h = 96, 54
    hh = h // 2 if upper else h  # upper half: where the hands are during an overhead clap
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                          f"scale={w}:{h},format=gray,crop={w}:{hh}:0:0", "-fps_mode",
                          "passthrough", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    n = w * hh
    fr = [raw[i:i + n] for i in range(0, len(raw) - n + 1, n)][:len(pts)]
    e = [0.0] + [sum(abs(a - b) for a, b in zip(f1, f0)) / n / max(t1 - t0, 1e-3)
                 for f0, f1, t0, t1 in zip(fr, fr[1:], pts, pts[1:])]
    return pts[:len(e)], e


def _grid(t, y, t0, t1):
    """y(t) linearly resampled to GRID_HZ on [t0, t1), z-scored."""
    out, j = [], 0
    for k in range(int((t1 - t0) * GRID_HZ)):
        x = t0 + k / GRID_HZ
        while j + 1 < len(t) and t[j + 1] <= x:
            j += 1
        if j + 1 < len(t) and t[j] <= x:
            a = (x - t[j]) / (t[j + 1] - t[j])
            out.append(y[j] * (1 - a) + y[j + 1] * a)
        else:
            out.append(y[min(j, len(y) - 1)])
    mu = sum(out) / len(out)
    sd = math.sqrt(sum((v - mu) ** 2 for v in out) / len(out)) or 1.0
    return [(v - mu) / sd for v in out]


def best_lag(ta, ya, tb, yb, t0, t1, lags):
    """(lag, r): lag in s maximising the correlation of a(t) with b(t + lag) on [t0, t1]."""
    a = _grid(ta, ya, t0, t1)
    best = None
    for lag in lags:
        b = _grid(tb, yb, t0 + lag, t1 + lag)
        r = sum(x * y for x, y in zip(a, b)) / len(a)
        if best is None or r > best[1]:
            best = (lag, r)
    return best


def mvnx_clap(m, dist, t_expected, window=0.5, max_dist=0.2):
    """Moment the hands meet in the clap nearest t_expected: first frame within 3 cm of the
    local wrist-distance minimum (hands often stay together for a moment after the clap)."""
    t = [f["t"] for f in m["frames"]]
    ks = [k for k in range(1, len(t) - 1) if abs(t[k] - t_expected) <= window
          and dist[k] <= dist[k - 1] and dist[k] < dist[k + 1] and dist[k] < max_dist]
    if not ks:
        return None
    k = kmin = min(ks, key=lambda k: dist[k])
    while k > 0 and dist[k - 1] < dist[kmin] + 0.03:
        k -= 1
    return t[k]


def sync_mvnx(m, video, pts, ts, te, max_spread=0.08, min_r=0.5):
    """Offset (t_xsens - t_video) and the two claps in Xsens time; ts/te = rough audio claps."""
    tv, ev = motion_series(video, pts)
    tm, sm = speed_series(m)
    g, gr = best_lag(tv, ev, tm, sm, 0, tv[-1], [k / 100 for k in range(-150, 151)])
    dist = hand_distance(m)
    c0, c1 = mvnx_clap(m, dist, ts + g), mvnx_clap(m, dist, te + g)
    if c0 is None or c1 is None:
        return None
    tu, eu = motion_series(video, pts, upper=True)
    th, sh = speed_series(m, HAND_SEGS)
    near = [g + k / 100 for k in range(-30, 31)]
    loc = [best_lag(tu, eu, th, sh, max(c - g - 1, 0), c - g + 1, near) for c in (c0, c1)]
    spread = abs(loc[0][0] - loc[1][0])
    if min(r for _, r in loc) >= min_r and spread <= max_spread:
        method, off = "motion_local", (loc[0][0] + loc[1][0]) / 2
    else:  # hands outside the upper image half or unclear: keep the whole-recording offset
        method, off = "motion_global", g
    return {"sync": method, "offset_s": round(off, 3), "offset_global_s": g,
            "motion_r": round(gr, 2), "offset_start_s": loc[0][0], "offset_end_s": loc[1][0],
            "offset_spread_ms": round(spread * 1000), "mvnx_start_s": c0, "mvnx_end_s": c1}


def write_xsens_csv(m, t0, t1, t_cut0, out):
    """Frames between t0..t1 (Xsens time of the first/last cut video frame); t_cut_s is on the cut video's clock (0 = first
    cut frame), so rows line up with the cut video by time."""
    header, cols = ["mvn_frame", "mvn_time_s", "t_cut_s"], []
    for field, (labels, comps) in EXPORT.items():
        if field in m["frames"][0]:
            cols.append(field)
            header += [f"{lab}_{c}" for lab in m[labels] for c in comps]
    half = 0.5 / m["rate"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(header)
        for f in m["frames"]:
            if t0 - half <= f["t"] <= t1 + half:
                w.writerow([f["index"], f["t"], round(f["t"] - t_cut0, 4)]
                           + [x for c in cols for x in f[c]])


def load_overrides(data):
    """clap_overrides.csv: nr;start_s;end_s  (e.g. 020;;22.461 -> keep auto start, set end)."""
    f = data / "clap_overrides.csv"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) == 3 and parts[0].isdigit():
            out[parts[0]] = [float(p) if p else None for p in parts[1:]]
    return out


def main():
    d = {k: Path(sys.argv[1]) / v for k, v in DIRS.items()}
    do_cut = "--cut" in sys.argv
    do_cfr = "--cfr30" in sys.argv
    outdir = d["sync"]
    sub = {"cut": d["cut"], "cfr": d["cfr"], "png": outdir / "Kontrollbilder", "csv": outdir / "Xsens_CSV"}
    outdir.mkdir(parents=True, exist_ok=True)
    for k in ["cut", "png"] * do_cut + ["cfr"] * do_cfr:
        sub[k].mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(d["base"])
    rows = []
    for v in sorted(d["video"].glob("*.mp4")):
        n = v.name[:3]
        pts = frame_times(v)
        claps, top, _ = find_claps(v)
        (ts, cs), (te, ce) = claps["start"], claps["end"]
        ov_s, ov_e = overrides.get(n, (None, None))
        if ov_s is not None:
            ts, cs = ov_s, "manual"
        if ov_e is not None:
            te, ce = ov_e, "manual"
        if ts is None or te is None:
            print(f"{n}  SKIPPED: no clap candidate, add it to clap_overrides.csv")
            continue
        row = {"file": v.name, "frames": len(pts), "audio_start_s": ts, "audio_start_conf": cs,
               "audio_end_s": te, "audio_end_conf": ce, "sync": "audio"}
        vs, ve, m, res = ts, te, None, None
        mvnx = next(iter(sorted(d["mvnx"].glob(n + "*.mvnx"))), None)
        if mvnx:
            m = read_mvnx(mvnx)
            res = sync_mvnx(m, v, pts, ts, te)
            if res is None:
                print(f"{n}  MVNX {mvnx.name}: no hand clap near the audio claps, cut by audio")
            else:
                row.update(res, mvnx_file=mvnx.name)
                vs, ve = res["mvnx_start_s"] - res["offset_s"], res["mvnx_end_s"] - res["offset_s"]
        cs_, ce_ = nearest_frame(pts, vs)[0], nearest_frame(pts, ve)[0]
        # The clap frame estimate is +-1 frame (checked visually), so cut one frame wider on
        # each side to keep the touching frame in the video.
        fs, fe = max(cs_ - 1, 0), min(ce_ + 1, len(pts) - 1)
        fs, fe = (fs, pts[fs]), (fe, pts[fe])
        row.update(clap_start_frame=cs_, clap_end_frame=ce_, cut_start_frame=fs[0],
                   cut_start_s=fs[1], cut_end_frame=fe[0], cut_end_s=fe[1])
        rows.append(row)
        extra = (f"offset {res['offset_s']:+.3f}s ({res['sync']}, start/end "
                 f"{res['offset_start_s']:+.2f}/{res['offset_end_s']:+.2f}, r {res['motion_r']})"
                 if res else "")
        print(f"{n}  cut {fs[1]:.3f}-{fe[1]:.3f}s (f{fs[0]}-f{fe[0]}, {row['sync']})  {extra}")
        stem = v.name.split(".mp4")[0]
        if do_cut:
            frame_strip(v, pts, fs[0], sub["png"] / f"{stem}_clapStart.png")
            frame_strip(v, pts, fe[0], sub["png"] / f"{stem}_clapEnd.png")
            cut(v, pts, fs[0], fe[0], sub["cut"] / f"{stem}_cut.mp4")
            if res:
                sub["csv"].mkdir(exist_ok=True)
                write_xsens_csv(m, fs[1] + res["offset_s"], fe[1] + res["offset_s"],
                                fs[1] + res["offset_s"], sub["csv"] / f"{mvnx.stem}_cut.csv")
        if do_cfr:
            cut(v, pts, fs[0], fe[0], sub["cfr"] / f"{stem}_cut_30fps.mp4", fps=30)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(outdir / "claps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", delimiter=";")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
