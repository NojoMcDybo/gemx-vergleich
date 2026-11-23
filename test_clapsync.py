"""Self-check for the Xsens side on synthetic data: python -I test_clapsync.py"""
import csv
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import clapsync as c

RATE, N = 60, 600  # 10 s
SEGS = ["Pelvis", "RightHand", "LeftHand"]


def hand_gap(t):
    # claps at 1.0 s and 8.0 s; hands stay together ~0.3 s after each clap; a box carried
    # 3-6 s keeps the hands 25 cm apart
    if 3 <= t <= 6:
        return 0.25
    for clap in (1.0, 8.0):
        if clap <= t <= clap + 0.3:
            return 0.02 + 0.01 * (t - clap)
        if clap - 0.3 < t < clap:
            return 0.02 + (clap - t) * 2  # closing at 2 m/s
    return 0.62


def write_mvnx(path):
    segs = "".join(f'<segment label="{s}" id="{k + 1}"/>' for k, s in enumerate(SEGS))
    frames = []
    for i in range(N):
        d = hand_gap(i / RATE)
        pos = " ".join(map(str, [0, 0, 1, d / 2, 0, 1.8, -d / 2, 0, 1.8]))
        frames.append(f'<frame time="{round(i * 1000 / RATE)}" index="{i}" type="normal">'
                      f'<orientation>{" ".join(["1 0 0 0"] * 3)}</orientation>'
                      f'<position>{pos}</position></frame>')
    path.write_text(
        '<?xml version="1.0"?><mvnx version="4" xmlns="http://www.xsens.com/mvn/mvnx">'
        f'<subject frameRate="{RATE}" segmentCount="3"><segments>{segs}</segments><frames>'
        '<frame time="0" type="tpose"><position>9 9 9 9 9 9 9 9 9</position></frame>'
        + "".join(frames) + "</frames></subject></mvnx>", encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "021_test.mvnx"
        write_mvnx(p)
        m = c.read_mvnx(p)
        assert m["rate"] == 60 and len(m["frames"]) == N  # tpose frame skipped
        dist = c.hand_distance(m)

        # clap = first frame the hands are within 3 cm of their closest point, not the
        # later minimum while they rest together; rough guess may be off by 0.4 s
        assert abs(c.mvnx_clap(m, dist, 1.4) - 1.0) <= 1 / RATE, c.mvnx_clap(m, dist, 1.4)
        assert abs(c.mvnx_clap(m, dist, 7.7) - 8.0) <= 1 / RATE
        assert c.mvnx_clap(m, dist, 4.5) is None  # carrying a box is not a clap

        # offset recovery: b is a delayed by 0.17 s
        ta = [k / 30 for k in range(300)]
        ya = [math.sin(x * 3) + (2 if 4 < x < 4.3 else 0) for x in ta]
        tb = [k / 60 for k in range(650)]
        yb = [math.sin((x - 0.17) * 3) + (2 if 4.17 < x < 4.47 else 0) for x in tb]
        lag, r = c.best_lag(ta, ya, tb, yb, 0.5, 9.0, [k / 100 for k in range(-50, 51)])
        assert abs(lag - 0.17) < 0.011 and r > 0.95, (lag, r)

        out = Path(tmp) / "out.csv"
        c.write_xsens_csv(m, 1.0, 8.0, 1.01, out)
        rows = list(csv.reader(out.open(encoding="utf-8"), delimiter=";"))
        assert rows[0][:5] == ["mvn_frame", "mvn_time_s", "t_cut_s", "Pelvis_x", "Pelvis_y"]
        assert "RightHand_q3" in rows[0]
        assert len(rows) - 1 == 421, len(rows)  # frames 60..480
        assert abs(float(rows[1][2]) + 0.01) < 1e-6, rows[1][:3]
    print("ok")


if __name__ == "__main__":
    main()
