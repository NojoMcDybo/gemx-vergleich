"""GemX-Vergleich: compare one Xsens MVNX with one GemX result in the browser.

Start: python vergleich_app.py [--port N] [--no-browser]   (needs the .venv-soma packages)
A local server (127.0.0.1 only) serves compare_view.html in app mode. It scans a data folder
(default OneDrive/Predimo, changeable in the page) for recordings: MVNX, GemX folder
(hpe_results.pt) and video are grouped by "<nr>_<date>" at the start of the name; results are
cached in %LOCALAPPDATA%/GemX-Vergleich. Files can also be dropped into the page; then the MVNX
and hpe_results.pt are uploaded and the video stays in the browser. GemX joints come from SOMA
(gemx_joints.py); the time offset Xsens <-> GemX is found by matching joint angles, so the
MVNX does not have to be cut first. GemX must have run on the constant-30-fps video.
"""
import hashlib
import io
import json
import shutil
import urllib.request
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import compare as c
from clapsync import read_mvnx

MAX_UPLOAD = 1 << 30
SYNC_STEP_S = 0.005
DATA_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "GemX-Vergleich"
LOG = DATA_DIR / "protokoll.txt"  # stdout/stderr when started without a console (pythonw)
IDLE_CHECKS = 18  # x 10 s without a request from the page: the window is closed, stop
APP_PORT = 47815  # fixed, because the page's localStorage (theme, stills, views) belongs to host + port
DEFAULT_ROOT = Path.home() / "OneDrive" / "Predimo"
KEY = re.compile(r"(\d{3})_(\d{8})_")
VIDEO_EXT = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime"}
gj = None  # gemx_joints (torch, ~4 s to import) is loaded in the background, the page comes first
layer, layer_ready, layer_lock, gemx = None, threading.Event(), threading.Lock(), {}
library = {}  # key -> {"mvnx": Path, "gemx": Path, "video": Path}, filled by scan()
HERE = Path(__file__).parent
REPO = "NojoMcDybo/gemx-vergleich"  # public, so releases are read without a login
UPDATE_HOSTS = {"api.github.com", "github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}
UPDATE_MAX = 50 << 20  # an update holds only the app files (~1 MB); Python and torch come with the setup


def angles_np(P):
    """(frames, joints, 3) -> (frames, len(ANGLES)) in degrees, like compare.angle."""
    out = []
    for a, b, p, q in c.ANGLES.values():
        u, v = P[:, c.IDX[b]] - P[:, c.IDX[a]], P[:, c.IDX[q]] - P[:, c.IDX[p]]
        cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12)
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.stack(out, 1)


def sample(t, Y, times):
    """Linear interpolation of Y (samples, ...) at times."""
    flat = Y.reshape(len(Y), -1)
    return np.stack([np.interp(times, t, flat[:, i]) for i in range(flat.shape[1])], 1).reshape(
        (len(times),) + Y.shape[1:])


def find_start(t, xa, ga, fps=c.FPS, step=SYNC_STEP_S):
    """(start_s, mean abs deg): Xsens time of GemX frame 0 whose joint angles fit best.
    Angles do not depend on the camera pose, so this works before any spatial alignment."""
    kt = np.arange(len(ga)) / fps
    starts = np.arange(t[0], t[-1] - kt[-1] + 1e-9, step)
    if not len(starts):
        raise ValueError(f"GemX ({kt[-1]:.1f} s) ist länger als die MVNX ({t[-1] - t[0]:.1f} s)")
    err = [np.abs(sample(t, xa, s + kt) - ga).mean() for s in starts]
    i = int(np.argmin(err))
    return float(starts[i]), float(err[i])


def xsens_positions(m):
    """MVNX dict from clapsync.read_mvnx -> (t, (frames, joints, 3) in compare.NAMES order)."""
    if not m["frames"] or "position" not in m["frames"][0]:
        raise ValueError("keine Segment-Positionen in der MVNX")
    seg = [m["segments"].index(s) for _, s in c.JOINTS]
    t = np.array([f["t"] for f in m["frames"]])
    pos = np.array([[f["position"][3 * i:3 * i + 3] for i in seg] for f in m["frames"]])
    return t, pos


def compare_record(m, gem, name):
    t, pos = xsens_positions(m)
    start, err = find_start(t, angles_np(pos), angles_np(gem))
    xs = sample(t, pos, start + np.arange(len(gem)) / c.FPS)
    frames = lambda a: [[tuple(p) for p in f] for f in a.tolist()]
    nr = re.match(r"\d+", name)
    rec = c.record(nr.group() if nr else "", name, None, frames(xs), frames(gem), "hpe_results.pt")
    rec["sync"] = {"xsens_start_s": round(start, 3), "angle_err_deg": round(err, 1)}
    return rec


def rec_key(name):
    """'028_20260825_BlockB_...' -> '028_20260825'; None for names without that start."""
    m = KEY.match(name)
    return m.group(1) + "_" + m.group(2) if m else None


def _rank(kind, p):
    # GemX ran on the cut constant-30-fps video, so only that one matches frame for frame
    n = p.stem.lower()
    v = (0 if n.endswith("_cut_30fps") else 2 if "incam" in n else 1) if kind == "video" else 0
    return v, len(str(p))


def scan(root):
    """{key: {"mvnx"|"gemx"|"video": Path}} for recordings in root that have MVNX or GemX."""
    found = {}
    for p in Path(root).rglob("*"):
        if p.name == "hpe_results.pt":
            key, kind = rec_key(p.parent.name), "gemx"
        elif p.suffix.lower() == ".mvnx":
            key, kind = rec_key(p.name), "mvnx"
        elif p.suffix.lower() in VIDEO_EXT:
            key, kind = rec_key(p.name), "video"
        else:
            continue
        if not key:
            continue
        old = found.setdefault(key, {}).get(kind)
        if old is None or _rank(kind, p) < _rank(kind, old):
            found[key][kind] = p
    return {k: v for k, v in sorted(found.items()) if "mvnx" in v or "gemx" in v}


def stem_of(paths):
    """File stem that names the recording: the MVNX, else the GemX result folder."""
    return paths["mvnx"].stem if "mvnx" in paths else paths["gemx"].parent.name


CACHE_VERSION = 5  # bump when the cached comparison gains or changes fields


def _sig(paths):
    return [CACHE_VERSION] + [[p.stat().st_size, p.stat().st_mtime_ns] for p in (paths["mvnx"], paths["gemx"])]


def card(rec):
    """Small summary for the library card (big numbers, sparkline of the posture error)."""
    s, pa = rec["summary"], rec["series"]["pa_mm"]
    n = 48
    spark = [round(max(pa[i * len(pa) // n:(i + 1) * len(pa) // n] or [0]), 1) for i in range(n)]
    return {"summary": s, "sync": rec.get("sync"), "spark": spark}


def cache_paths(key):
    return DATA_DIR / "cache" / f"{key}.json", DATA_DIR / "cache" / f"{key}.card.json"


def cached_card(key, paths, strict=False):
    """Card of a cached comparison of these files. Not strict: a card from an older CACHE_VERSION
    still counts (the card format has not changed), so a version bump keeps the library compared;
    the full comparison is recomputed when it is opened."""
    _, cp = cache_paths(key)
    try:
        d, sig = json.loads(cp.read_text(encoding="utf-8")), _sig(paths)
        return d["card"] if d["sig"] == sig or (not strict and d["sig"][1:] == sig[1:]) else None
    except (OSError, ValueError, KeyError):
        return None


def get_rec(key):
    """Comparison for a library recording as JSON text, from the cache or computed and cached."""
    paths = library.get(key)
    if not paths or "mvnx" not in paths or "gemx" not in paths:
        raise ValueError("Für diese Aufnahme fehlt die MVNX oder das GemX-Ergebnis")
    full, cp = cache_paths(key)
    if cached_card(key, paths, strict=True) is not None and full.exists():
        return full.read_text(encoding="utf-8")
    j = joints_from(paths["gemx"])
    rec = compare_record(read_mvnx(paths["mvnx"]), j, paths["mvnx"].stem)
    body = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(body, encoding="utf-8")
    cp.write_text(json.dumps({"sig": _sig(paths), "card": card(rec)}), encoding="utf-8")
    return body


def joints_from(src):
    layer_ready.wait()
    if layer is None:
        raise RuntimeError(f"SOMA-Modell konnte nicht geladen werden, siehe {LOG}")
    try:
        with layer_lock:
            return gj.load_joints(layer, src).numpy().astype(float)
    except Exception as e:  # torch's own message suggests unsafe loading, so replace it
        raise ValueError("keine lesbare GemX-Ergebnisdatei (hpe_results.pt erwartet)") from e


def settings():
    try:
        return json.loads((DATA_DIR / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def root_dir():
    r = Path(settings().get("root") or DEFAULT_ROOT)
    return r if r.is_dir() else None


def set_root(path):
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f"Ordner nicht gefunden: {path}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "settings.json").write_text(json.dumps({"root": str(p)}), encoding="utf-8")


def library_json():
    root = root_dir()
    found = scan(root) if root else {}
    library.clear()
    library.update(found)
    items = [{"key": k, "nr": k[:3], **dict(zip(("name", "view", "date"), c.display_name(stem_of(v)))),
              "has": {t: t in v for t in ("mvnx", "gemx", "video")},
              "size": {t: p.stat().st_size for t, p in v.items()},
              "card": cached_card(k, v) if "mvnx" in v and "gemx" in v else None}
             for k, v in found.items()]
    return json.dumps({"root": str(root) if root else None, "items": items}, ensure_ascii=False)


PARTS = ("report", "csv", "html", "video", "raw")


def _safe(name):
    return re.sub(r'[<>:"/\|?*\x00-\x1f]', "_", str(name)).strip(" .")[:120] or "Aufnahme"


def build_export(req, out):
    """Write the share package to the open binary file out. req: {"parts": [...], "items":
    [{"key": library key | "rec": comparison posted by the page, "report": text}]}.
    Videos and raw files exist only for library recordings (the server knows their paths)."""
    parts = set(req.get("parts") or []) & set(PARTS)
    if not parts or not req.get("items"):
        raise ValueError("nichts ausgewählt")
    recs = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for it in req["items"]:
            key = it.get("key")
            paths = library.get(key, {}) if key else {}
            if key and not paths:
                raise ValueError(f"Aufnahme {key} nicht mehr im Ordner")
            rec = json.loads(get_rec(key)) if paths else it.get("rec")
            if not isinstance(rec, dict) or "xs" not in rec:
                raise ValueError("Vergleichsdaten fehlen")
            title = _safe(f"{rec.get('nr', '')} {rec.get('name', '')}")
            if "report" in parts and it.get("report"):
                z.writestr(f"Auswertung/{title}.txt", str(it["report"]))
            video = paths.get("video") if "video" in parts else None
            if video:
                z.write(video, f"Videos/{video.name}", compress_type=zipfile.ZIP_STORED)
            if "raw" in parts and paths:
                for kind in ("mvnx", "gemx"):
                    z.write(paths[kind], f"Rohdaten/{key}/{paths[kind].name}")
            recs.append(dict(rec, video=f"Videos/{video.name}" if video else None, aufnahme=rec.get("stem") or title))
        if "csv" in parts:
            buf = io.StringIO()
            c.write_table(buf, [c.table_row(r, r["aufnahme"]) for r in recs])
            z.writestr("Kennzahlen.csv", buf.getvalue().encode("utf-8-sig"))  # BOM: Excel reads the umlauts
        if "html" in parts:
            z.writestr("Vergleich.html", c.page(dict(c.meta(), recordings=recs)))
    return len(recs)


def export_name(n):
    return f"GemX-Vergleich_{n}_Aufnahme{'n' if n != 1 else ''}_{time.strftime('%Y-%m-%d_%H%M')}.zip"


def save_to_downloads(req):
    """Build the package in the Downloads folder and show it selected in the Explorer."""
    folder = Path.home() / "Downloads"
    folder.mkdir(exist_ok=True)
    tmp = folder / f".gemx-export-{uuid.uuid4().hex}.part"
    try:
        with open(tmp, "wb") as fh:
            n = build_export(req, fh)
        dest = folder / export_name(n)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    subprocess.Popen(["explorer", "/select,", str(dest)])
    return dest


PICK_PS = r"""Add-Type -AssemblyName System.Windows.Forms
$f = New-Object System.Windows.Forms.Form -Property @{TopMost=$true; ShowInTaskbar=$false}
$d = New-Object System.Windows.Forms.FolderBrowserDialog
$d.Description = 'Ordner mit den Aufnahmen wählen'
if ($d.ShowDialog($f) -eq 'OK') { [Console]::OutputEncoding = [Text.Encoding]::UTF8; $d.SelectedPath }"""


def pick_folder():
    """Native folder dialog (Windows); '' when cancelled."""
    r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", PICK_PS], capture_output=True,
                       timeout=600, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.stdout.decode("utf-8", "replace").strip()


def _load_layer():
    global layer, gj
    try:
        import gemx_joints as gj
        assert list(gj.SOMA_JOINTS) == c.NAMES
        layer = gj.make_layer()
    finally:  # on failure the traceback stays in the console (or the log) and uploads get an error
        layer_ready.set()


# ---- Updates from the GitHub releases of REPO ----
# A release carries update.json ({"version", "runtime", "zip", "sha256"}), the app zip and the
# setup. The zip replaces only the files in this folder; "runtime" fingerprints the Python
# packages, so a release that needs other packages points to its setup instead.

def app_version():
    """version.json written by build_bundle.py; None when run from the source folder (no updates)."""
    try:
        return json.loads((HERE / "version.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _vkey(v): return tuple(int(x) for x in re.findall(r"\d+", v))


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)  # GitHub sends downloads to its asset host; nowhere else
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _check_url(url):
    u = urlparse(url)
    if u.scheme != "https" or u.hostname not in UPDATE_HOSTS:
        raise ValueError(f"Update-Adresse nicht erlaubt: {u.hostname}")


def _fetch(url, limit, timeout=10):
    _check_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": "GemX-Vergleich", "Accept": "*/*"})
    with urllib.request.build_opener(_SafeRedirect).open(req, timeout=timeout) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Update-Datei zu groß")
    return data


def check_update():
    """What the latest release offers compared with this installation."""
    cur = app_version()
    if not cur:
        return {"current": None, "available": False}
    rel = json.loads(_fetch(f"https://api.github.com/repos/{REPO}/releases/latest", 1 << 20))
    assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
    if "update.json" not in assets:
        return {"current": cur["version"], "available": False}
    info = json.loads(_fetch(assets["update.json"], 1 << 16))
    newer = _vkey(info["version"]) > _vkey(cur["version"])
    return {"current": cur["version"], "latest": info["version"], "available": newer,
            "setup_only": newer and info.get("runtime") != cur.get("runtime"),
            "notes": rel.get("body") or "", "page": rel.get("html_url"),
            "zip": assets.get(info.get("zip", "")), "sha256": info.get("sha256")}


def install_update(st):
    """Download, check and copy the app files of the release described by st (check_update()).
    The replaced files go to DATA_DIR/update-backup/<old version>; a failed copy is undone."""
    if not st.get("available") or st.get("setup_only") or not st.get("zip"):
        raise ValueError("Kein Update zum Installieren")
    data = _fetch(st["zip"], UPDATE_MAX, timeout=60)
    if hashlib.sha256(data).hexdigest() != st["sha256"]:
        raise ValueError("Prüfsumme des Updates stimmt nicht, nichts geändert")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        for n in names:
            if n.startswith(("/", "\\")) or ":" in n or ".." in Path(n).parts:
                raise ValueError(f"Ungültiger Pfad im Update: {n}")
        if not {"vergleich_app.py", "version.json"} <= set(names):
            raise ValueError("Update unvollständig")
        if json.loads(z.read("version.json")).get("version") != st["latest"]:
            raise ValueError("Update passt nicht zur Version des Releases")
        files = {n: z.read(n) for n in names}
    backup = DATA_DIR / "update-backup" / st["current"]
    done = []
    try:
        for n, content in files.items():
            target = HERE / n
            if target.exists():
                (backup / n).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup / n)
            target.parent.mkdir(parents=True, exist_ok=True)
            done.append(n)
            target.write_bytes(content)
    except OSError:
        for n in done:  # put back what was already replaced
            if (backup / n).exists():
                shutil.copy2(backup / n, HERE / n)
            else:
                (HERE / n).unlink(missing_ok=True)
        raise
    return len(files)


def restart(srv):
    """Start this program again with the new files; the open window reloads once it answers.
    shortcut: the new process cannot wait for the old Edge window, so after an update it ends
    through the watchdog (~3 min after the window closes) instead of at once."""
    time.sleep(0.5)  # let the answer reach the page
    srv.shutdown()
    srv.server_close()
    (DATA_DIR / "server.json").unlink(missing_ok=True)
    args = [a for a in sys.orig_argv[1:] if a != "--after-update"]
    subprocess.Popen([sys.executable, *args, "--after-update"], cwd=str(HERE))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _local(self):
        # Host blocks DNS rebinding; Origin/Sec-Fetch-Site block other web pages calling the API
        own = f"127.0.0.1:{self.server.server_port}"
        ok = (self.headers.get("Host") == own
              and self.headers.get("Origin", "http://" + own) == "http://" + own
              and self.headers.get("Sec-Fetch-Site", "same-origin") in ("same-origin", "none"))
        if not ok:
            self._send(403, "forbidden", "text/plain")
        return ok

    def _error(self, what, e):  # shown in the page; the server keeps running
        msg = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
        self._send(400, f"{what}: {msg}", "text/plain; charset=utf-8")

    def do_GET(self):
        if not self._local():
            return
        seen[0] = 0
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/ping":
                return self._send(200, json.dumps({"app": "gemx-vergleich", "version": (app_version() or {}).get("version")}))
            if url.path == "/api/update":
                return self._send(200, json.dumps(check_update(), ensure_ascii=False))
            if url.path == "/":
                page = c.page(dict(c.meta(), recordings=[], app=True))
                return self._send(200, page, "text/html; charset=utf-8")
            if url.path == "/api/library":
                return self._send(200, library_json())
            if url.path == "/api/rec":
                return self._send(200, get_rec(q.get("key", "")))
            if url.path == "/api/video":
                return self._video(library.get(q.get("key", ""), {}).get("video"))
            if url.path == "/api/pick-root":
                path = pick_folder()
                if path:
                    set_root(path)
                return self._send(200, json.dumps({"picked": bool(path)}))
            self._send(404, "not found", "text/plain")
        except Exception as e:
            self._error("Aufnahme", e)

    def _video(self, path):
        """Serve a library video with HTTP Range support, so the page can seek in it."""
        if path is None:
            return self._send(404, "kein Video", "text/plain; charset=utf-8")
        size = path.stat().st_size
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        start, end = 0, size - 1
        if m and (m[1] or m[2]):
            if m[1]:
                start, end = int(m[1]), min(int(m[2] or size - 1), size - 1)
            else:
                start = max(0, size - int(m[2]))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if m else 200)
        self.send_header("Content-Type", VIDEO_EXT[path.suffix.lower()])
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if m:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except OSError:  # the browser often aborts video requests while seeking
            pass

    def do_POST(self):
        if not self._local():
            return
        seen[0] = 0
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_UPLOAD:
            return self._send(413, "Datei leer oder zu groß", "text/plain; charset=utf-8")
        body = io.BytesIO(self.rfile.read(n))
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        what = "Anfrage"
        try:
            if url.path == "/api/root":
                what = "Ordner"
                set_root(json.loads(body.getvalue().decode("utf-8"))["path"])
                return self._send(200, library_json())
            if url.path == "/api/export":
                what = "Teilen"
                req = json.loads(body.getvalue().decode("utf-8"))
                if q.get("to") == "downloads":
                    return self._send(200, json.dumps({"path": str(save_to_downloads(req))}))
                with tempfile.TemporaryFile() as fh:
                    n = build_export(req, fh)
                    size = fh.tell()
                    fh.seek(0)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Length", str(size))
                    self.send_header("Content-Disposition", f'attachment; filename="{export_name(n)}"')
                    self.end_headers()
                    while chunk := fh.read(1 << 20):
                        self.wfile.write(chunk)
                return
            if url.path == "/api/update":
                what = "Update"
                n = install_update(check_update())  # fresh, not what the page was shown
                threading.Thread(target=restart, args=(self.server,)).start()
                return self._send(200, json.dumps({"files": n}))
            if url.path == "/api/gemx":
                what = "GemX-Datei"
                j = joints_from(body)
                key = uuid.uuid4().hex
                gemx[key] = j
                return self._send(200, json.dumps({"id": key, "frames": len(j)}))
            if url.path == "/api/compare":
                what = "MVNX-Datei"
                gem = gemx.pop(q.get("gemx", ""), None)
                if gem is None:
                    raise ValueError("GemX-Daten fehlen, bitte erneut vergleichen")
                rec = compare_record(read_mvnx(body), gem, q.get("name", "Aufnahme"))
                return self._send(200, json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
            self._send(404, "not found", "text/plain")
        except Exception as e:
            self._error(what, e)

    def log_message(self, *args):
        pass


seen = [0]  # watchdog ticks since the page last called


def running_url():
    """URL of a GemX-Vergleich that is already running: a second start opens it instead."""
    try:
        url = json.loads((DATA_DIR / "server.json").read_text(encoding="utf-8"))["url"]
        import urllib.request
        with urllib.request.urlopen(url + "api/ping", timeout=1.5) as r:
            return url if json.loads(r.read()).get("app") == "gemx-vergleich" else None
    except Exception:
        return None


class Server(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"  # on Windows SO_REUSEADDR lets a second program bind the same port


def open_window(url):
    """Show the page in its own window (Edge app mode with an own profile, so no tabs or address bar
    and not mixed into the user's Edge). Returns the Edge process, which ends with its last window;
    None when the page went to the default browser instead."""
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        exe = Path(base or ".") / "Microsoft" / "Edge" / "Application" / "msedge.exe"
        if base and exe.exists():
            profile = DATA_DIR / "fenster"
            # the first window fills the screen (Edge's default is ~1000 px, too narrow); later Edge reuses the last size
            first = [] if profile.exists() else ["--start-maximized"]
            show = subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW, wShowWindow=1)  # SW_SHOWNORMAL
            return subprocess.Popen([str(exe), f"--app={url}", f"--user-data-dir={profile}", "--no-first-run",
                                     "--no-default-browser-check", *first], startupinfo=show)
    webbrowser.open(url)
    return None


def close_with(proc, srv):
    """Stop the server as soon as the app window's Edge process is gone."""
    proc.wait()
    srv.shutdown()


def watchdog(srv):
    """Stop when the page has not called for IDLE_CHECKS ticks (it pings every 20 s). Counts
    ticks, not clock time, so a PC waking from sleep does not end the server under the page."""
    while True:
        time.sleep(10)
        seen[0] += 1
        if seen[0] > IDLE_CHECKS:
            srv.shutdown()
            return


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    console = sys.stdout is not None
    if not console:  # pythonw (installed shortcut): no console, write to the log instead
        sys.stdout = sys.stderr = open(LOG, "w", encoding="utf-8", buffering=1)
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 0
    updated = "--after-update" in sys.argv  # restarted by an update: the old window is still open
    if not port and not updated and (url := running_url()):
        open_window(url)
        return
    threading.Thread(target=_load_layer, daemon=True).start()
    srv = None
    for _ in range(40 if updated else 1):  # after an update the old process may still hold the port
        try:
            srv = Server(("127.0.0.1", port or APP_PORT), Handler)
            break
        except OSError:
            if updated:
                time.sleep(0.25)
    if srv is None:
        if port:
            raise OSError(f"Port {port} ist belegt")
        # shortcut: APP_PORT is taken by another program, so this session's page settings start empty
        srv = Server(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{srv.server_port}/"
    if not port:
        (DATA_DIR / "server.json").write_text(json.dumps({"url": url}), encoding="utf-8")
    print(f"GemX-Vergleich läuft unter {url}" + ("\nZum Beenden dieses Fenster schließen." if console else ""), flush=True)
    if "--no-browser" not in sys.argv:
        if not updated and (proc := open_window(url)):
            threading.Thread(target=close_with, args=(proc, srv), daemon=True).start()
        threading.Thread(target=watchdog, args=(srv,), daemon=True).start()  # backstop, e.g. Edge kept running
    srv.serve_forever()


if __name__ == "__main__":
    main()
