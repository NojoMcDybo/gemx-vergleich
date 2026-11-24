"""Self-check for vergleich_app.py (needs .venv-soma): .venv-soma\\Scripts\\python test_vergleich_app.py"""
import json
import math
from pathlib import Path

import numpy as np

import compare as c
import test_compare as tc
import vergleich_app as va

RATE, START = 60, 1.437  # Xsens rate, Xsens time of GemX frame 0


def main():
    # Xsens: 8 s at 60 Hz as a fake MVNX dict (all segments positions, JOINTS order + extras)
    segs = ["Extra"] + [s for _, s in c.JOINTS]
    frames = []
    for i in range(8 * RATE):
        t = 0.2 + i / RATE
        pos = [0.0, 0.0, 0.0] + [v for p in tc.xsens_frame(t * c.FPS) for v in p]
        frames.append({"t": t, "position": pos})
    m = {"segments": segs, "frames": frames}

    # GemX: 4 s at 30 fps starting at START, rotated/shifted like a camera view, 3 mm noise
    rnd = np.random.default_rng(1)
    T = (1.0, tc.rotation((0.3, -1, 0.5), 140), (0.25, -0.04, 3.0))
    gem = np.array([[c._apply(T, p) for p in tc.xsens_frame((START + k / c.FPS) * c.FPS)]
                    for k in range(4 * c.FPS)]) + rnd.normal(0, 0.003, (4 * c.FPS, len(c.NAMES), 3))

    rec = va.compare_record(m, gem, "042_test_cut_30fps")
    assert abs(rec["sync"]["xsens_start_s"] - START) <= va.SYNC_STEP_S, rec["sync"]
    s = rec["summary"]
    assert s["lag_frames"] == 0 and s["mpjpe_mm"] < 10 and s["pa_mpjpe_mm"] < 6, s
    assert rec["nr"] == "042" and len(rec["xs"]) == len(gem)
    # camera (origin, +z of the GemX space) seen from Xsens: inverse of the test transform
    R, t = T[1], T[2]
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    want_pos, want_dir = c._rot(Rt, tuple(-v for v in t)), c._rot(Rt, (0, 0, 1))
    cam = rec["series"]["cam"]
    assert all(abs(a - b) < 0.02 for a, b in zip(cam["pos"], want_pos)), (cam, want_pos)
    assert all(abs(a - b) < 0.01 for a, b in zip(cam["dir"], want_dir)), (cam, want_dir)

    try:
        va.find_start(np.arange(0, 1, 1 / RATE), np.zeros((RATE, 8)), np.zeros((40, 8)))
        raise AssertionError("GemX longer than MVNX must fail")
    except ValueError:
        pass
    assert math.isclose(va.angles_np(np.array([tc.xsens_frame(5)]))[0, 4],
                        c.angle(tc.xsens_frame(5), *c.ANGLES["Ellbogen R"]), abs_tol=1e-9)
    library_and_video(rec)
    updates()
    groups()
    print("ok")


def library_and_video(rec):
    """Folder scan groups files by '<nr>_<date>'; the video endpoint honours Range and the
    API refuses requests from other web pages."""
    import tempfile, threading, urllib.error, urllib.request
    assert c.display_name("028_20260825_BlockB_W_X_GTV_GoPro_Vorne_1080p30fps_BoxhintenAusfallschrittL")         == ("Boxhinten Ausfallschritt L", "Vorne", "25.08.2026")
    assert va.rec_key("028_20260825_x.mvnx") == "028_20260825" and va.rec_key("hpe_results.pt") is None
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        files = {"Xsens/007_20260101_W_X_Seite_Sprung.mvnx": b"m",
                 "GemX/007_20260101_W_G_Seite_Sprung_cut_30fps/hpe_results.pt": b"g",
                 "Videos/007_20260101_W_G_Seite_Sprung.mp4": b"original",
                 "Videos/cut/007_20260101_W_G_Seite_Sprung_cut_30fps.mp4": bytes(range(100)),
                 "Videos/008_20260101_only_video.mp4": b"v", "notes.txt": b"x"}
        for n, data in files.items():
            (d / n).parent.mkdir(parents=True, exist_ok=True)
            (d / n).write_bytes(data)
        lib = va.scan(d)
        assert list(lib) == ["007_20260101"], lib  # 008 has neither MVNX nor GemX
        assert lib["007_20260101"]["video"].name.endswith("_cut_30fps.mp4")
        assert va.stem_of(lib["007_20260101"]) == "007_20260101_W_X_Seite_Sprung"
        # a card from an older cache version stays on the library card, but the full comparison is redone
        data_dir, va.DATA_DIR = va.DATA_DIR, d / "appdata"
        try:
            _, cp = va.cache_paths("007_20260101")
            cp.parent.mkdir(parents=True)
            old = [va.CACHE_VERSION - 1] + va._sig(lib["007_20260101"])[1:]
            cp.write_text(json.dumps({"sig": old, "card": {"x": 1}}), encoding="utf-8")
            assert va.cached_card("007_20260101", lib["007_20260101"]) == {"x": 1}
            assert va.cached_card("007_20260101", lib["007_20260101"], strict=True) is None
        finally:
            va.DATA_DIR = data_dir

        va.library.clear(); va.library.update(lib)
        srv = va.ThreadingHTTPServer(("127.0.0.1", 0), va.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        own = f"127.0.0.1:{srv.server_port}"

        def get(path, **headers):
            req = urllib.request.Request(f"http://{own}{path}", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, r.headers, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.headers, e.read()
        st, h, body = get("/api/video?key=007_20260101", Range="bytes=10-19")
        assert st == 206 and body == bytes(range(10, 20)) and h["Content-Range"] == "bytes 10-19/100", (st, h)
        st, h, body = get("/api/video?key=007_20260101", Range="bytes=-5")
        assert st == 206 and body == bytes(range(95, 100))
        st, _, body = get("/api/video?key=007_20260101")
        assert st == 200 and len(body) == 100
        assert get("/api/video?key=007_20260101", Range="bytes=200-")[0] == 416
        assert get("/api/video?key=999_20260101")[0] == 404
        assert get("/api/video?key=007_20260101", Origin="http://evil.example")[0] == 403
        assert get("/api/video?key=007_20260101", **{"Sec-Fetch-Site": "cross-site"})[0] == 403
        # second start finds the running program through server.json + /api/ping
        data_dir, va.DATA_DIR = va.DATA_DIR, d / "appdata"
        try:
            assert va.running_url() is None or (va.DATA_DIR / "server.json").exists()
            (va.DATA_DIR / "server.json").write_text(json.dumps({"url": f"http://{own}/"}), encoding="utf-8")
            assert va.running_url() == f"http://{own}/"
            va.seen[0] = 5; get("/api/ping"); assert va.seen[0] == 0  # every page call resets the idle count
        finally:
            va.DATA_DIR = data_dir
        srv.shutdown()
        data_dir, va.DATA_DIR = va.DATA_DIR, d / "appdata"
        try:
            assert va.running_url() is None  # nothing answers any more
        finally:
            va.DATA_DIR = data_dir

        # share package: library recordings bring video + raw files, posted ones only their data
        import io, zipfile
        get_rec, va.get_rec = va.get_rec, lambda key: json.dumps(rec)
        try:
            buf = io.BytesIO()
            n = va.build_export({"parts": list(va.PARTS), "items": [{"key": "007_20260101", "report": "Text"},
                                                                     {"rec": dict(rec, nr="043", name="a/b"), "report": "Zwei"}]}, buf)
            names = zipfile.ZipFile(buf).namelist()
            assert n == 2 and set(names) == {
                f"Auswertung/042 {rec['name']}.txt", "Auswertung/043 a_b.txt",
                "Videos/007_20260101_W_G_Seite_Sprung_cut_30fps.mp4",
                "Rohdaten/007_20260101/007_20260101_W_X_Seite_Sprung.mvnx", "Rohdaten/007_20260101/hpe_results.pt",
                "Kennzahlen.csv", "Vergleich.html"}, names
            html = zipfile.ZipFile(buf).read("Vergleich.html").decode("utf-8")
            assert '"video":"Videos/007_20260101_W_G_Seite_Sprung_cut_30fps.mp4"' in html
            for bad in ({"parts": [], "items": [{"key": "007_20260101"}]}, {"parts": ["csv"], "items": [{"key": "999_x"}]},
                        {"parts": ["csv"], "items": [{"rec": {"no": 1}}]}):
                try:
                    va.build_export(bad, io.BytesIO())
                    raise AssertionError(bad)
                except ValueError:
                    pass
        finally:
            va.get_rec = get_rec
        assert va._safe('a/b:c\x01 . ') == "a_b_c_"


def updates():
    """Version order, allowed hosts, and installing a release zip into a scratch app folder:
    checksum, version and paths are checked, replaced files are backed up."""
    import hashlib, io, tempfile, zipfile
    assert va._vkey("2026.11.23.1530") > va._vkey("2026.11.23.930") > va._vkey("2026.11.9.2359")
    for bad in ("http://github.com/x", "https://example.com/x", "https://github.com.evil.de/x"):
        try:
            va._check_url(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    va._check_url("https://objects.githubusercontent.com/x")

    def zipped(files):
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            for n, t in files.items():
                z.writestr(n, t)
        return b.getvalue()

    here, data_dir, fetch = va.HERE, va.DATA_DIR, va._fetch
    with tempfile.TemporaryDirectory() as tmp:
        va.HERE, va.DATA_DIR = Path(tmp) / "app", Path(tmp) / "data"
        va.HERE.mkdir()
        (va.HERE / "vergleich_app.py").write_text("old")
        (va.HERE / "version.json").write_text('{"version": "2026.1.1.1000"}')
        good = zipped({"vergleich_app.py": "new", "compare.py": "c", "version.json": '{"version": "2026.2.1.1000"}'})
        st = {"available": True, "setup_only": False, "zip": "https://github.com/z", "current": "2026.1.1.1000",
              "latest": "2026.2.1.1000", "sha256": hashlib.sha256(good).hexdigest()}
        try:
            for payload, sha, why in ((good, "0" * 64, "checksum"),
                                      (zipped({"../x.py": "", "vergleich_app.py": "", "version.json": "{}"}), None, "path"),
                                      (zipped({"vergleich_app.py": "", "version.json": '{"version": "1"}'}), None, "version")):
                va._fetch = lambda url, limit, timeout=10, d=payload: d
                try:
                    va.install_update(dict(st, sha256=sha or hashlib.sha256(payload).hexdigest()))
                    raise AssertionError(why)
                except ValueError:
                    pass
            assert (va.HERE / "vergleich_app.py").read_text() == "old" and not (Path(tmp) / "x.py").exists()
            va._fetch = lambda url, limit, timeout=10: good
            assert va.install_update(st) == 3
            assert (va.HERE / "vergleich_app.py").read_text() == "new" and (va.HERE / "compare.py").exists()
            assert (va.DATA_DIR / "update-backup" / "2026.1.1.1000" / "vergleich_app.py").read_text() == "old"
            assert va.app_version()["version"] == "2026.2.1.1000"
            for blocked in ({"setup_only": True}, {"available": False}):
                try:
                    va.install_update(dict(st, **blocked))
                    raise AssertionError(blocked)
                except ValueError:
                    pass
        finally:
            va.HERE, va.DATA_DIR, va._fetch = here, data_dir, fetch


def groups():
    import tempfile
    data_dir = va.DATA_DIR
    with tempfile.TemporaryDirectory() as tmp:
        va.DATA_DIR = Path(tmp) / "data"
        try:
            assert va.load_groups() == {"groups": [], "members": {}}
            out = va.save_groups({"groups": [{"id": "g1", "name": "  Frontal ", "color": 2}],
                                  "members": {"042_20250101": ["g1", "weg"], "043_20250101": ["weg"]}})
            assert out == {"groups": [{"id": "g1", "name": "Frontal", "color": 2}], "members": {"042_20250101": ["g1"]}}
            assert va.load_groups() == out  # unknown group ids and empty members are dropped
            for bad in ([], {"groups": {}, "members": {}}, {"groups": [{"id": "", "name": "x", "color": 0}], "members": {}}):
                try:
                    va.save_groups(bad)
                    raise AssertionError(f"accepted {bad}")
                except ValueError:
                    pass
        finally:
            va.DATA_DIR = data_dir


if __name__ == "__main__":
    main()
