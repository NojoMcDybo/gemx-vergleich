"""Build the GemX-Vergleich installer (and optionally a portable zip) for colleagues.

Run: python build_bundle.py [--zip] [--release ["notes"]]
  -> dist/GemX-Vergleich/ (portable folder) and dist/GemX-Vergleich-Setup.exe (Inno Setup 6,
     installer.iss; ISCC.exe must be installed: winget install JRSoftware.InnoSetup)
  --zip also writes dist/GemX-Vergleich.zip.
  --release also writes the update package (app files only) and publishes setup + package as a
     GitHub release of vergleich_app.REPO (gh must be logged in); the installed programs offer it
     as an update. The text after --release becomes the release notes shown in the program.
Copies the local Python 3.11 install (base of .venv-soma), the venv packages, the app files
and the SOMA assets. Nothing else is needed on the target PC (Windows x64).
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
VENV = HERE / ".venv-soma"
SITE = VENV / "Lib" / "site-packages"
# the Python install the venv was made from (its "home"), copied into the bundle
PY = Path(next(line.split("=", 1)[1].strip() for line in (VENV / "pyvenv.cfg").read_text().splitlines()
               if line.startswith("home")))
OUT = HERE / "dist" / "GemX-Vergleich"
APP_FILES = ["vergleich_app.py", "compare.py", "compare_view.html", "nojo-ui.css", "nojo-ui.js",
             "gemx_joints.py", "clapsync.py", "gemx.ico"]
# Visual C++ runtime DLLs torch needs; app-local copies so PCs without the redistributable work
VC_DLLS = ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll", "vcomp140.dll"]
# Not needed to run: C/C++ headers (torch alone ~9500 files), package test suites, link
# libraries and debug symbols. Halves the file count, which is what makes installing slow.
SKIP_SITE = shutil.ignore_patterns("pip", "pip-*", "setuptools", "setuptools-*", "_distutils_hack",
                                   "distutils-precedence.pth", "__pycache__", "include", "tests",
                                   "*.lib", "*.pdb", "*.whl")
REPO = "NojoMcDybo/gemx-vergleich"  # same as vergleich_app.REPO (not imported: that pulls in numpy)
ISCC = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe"]

START = r"""@echo off
title GemX-Vergleich
"%~dp0python\python.exe" -E -s "%~dp0app\vergleich_app.py"
pause
"""
README = """GemX-Vergleich
==============

Vergleicht eine Xsens-Aufnahme (MVNX) mit dem GemX-Ergebnis (hpe_results.pt) derselben
Bewegung: Skelett über dem Video, Fehler pro Frame, Gelenkwinkel, Seitenvertauschungen.

Start: über "GemX-Vergleich" im Startmenü oder auf dem Desktop (nach der Installation), bzw. in
der Zip-Fassung per Doppelklick auf "GemX-Vergleich starten.bat". Die Ansicht öffnet sich in
einem eigenen Fenster (Microsoft Edge im App-Modus). Schließen des Fensters beendet das Programm;
ein zweiter Start öffnet ein weiteres Fenster des laufenden Programms. Meldungen stehen in
%LOCALAPPDATA%\\GemX-Vergleich\\protokoll.txt.

Aufnahmen finden: Oben rechts den Ordner mit den Aufnahmen wählen (Unterordner werden mit
durchsucht). MVNX, GemX-Ergebnisordner und Video gehören zusammen, wenn ihr Name gleich beginnt,
z. B. "028_20260825_...". Jede Aufnahme bekommt eine Karte; ein Klick vergleicht und öffnet sie.
Ergebnisse werden in %LOCALAPPDATA%\\GemX-Vergleich zwischengespeichert.
Oder Dateien bzw. einen Ordner in das Feld oben ziehen: MVNX, hpe_results.pt und optional das
30-fps-Video, auf dem GemX lief. Den Zeitversatz
zwischen Xsens und GemX findet das Programm selbst über die Gelenkwinkel (MVNX muss nicht
geschnitten sein, die GemX-Aufnahme muss aber ganz in der MVNX liegen).
Alles läuft lokal auf diesem PC; Messdaten werden nicht verschickt (nur die Update-Abfrage geht
an GitHub).

Auswertung: Simulation groß, Video klein. Griff oben antippen = Anordnung wählen (auch 70 : 30),
ziehen: an den linken oder rechten Rand = nebeneinander, sonst Bild im Bild. Steg zwischen den
Fenstern bzw. Ecke des kleinen Fensters ziehen ändert die Größe. Simulation: ziehen dreht frei,
kurz tippen spielt ab, Würfel unten links wählt die Ansicht, Kamera-Knopf zeigt den Blick der
Videokamera. Video: Mausrad zoomt. Gelenk in der Figur unten links oder in der Simulation
antippen zeigt seinen Verlauf. Das Fragezeichen in der Kennzahlenkarte erklärt jedes Element.
Übersicht und Auswertung wechseln auch per Wischen mit gedrückter Maustaste (nach links zur
Auswertung, nach rechts zurück). Sonne/Mond oben rechts schaltet hell/dunkel.
Teilen (in der Übersicht): von nur dem Text bis zum Paket mit Videos und Rohdaten.
Leertaste spielt ab, Pfeiltasten gehen Frame für Frame, Esc führt zurück.

Kennzahlen:
- MPJPE: mittlerer Gelenkabstand, nachdem GemX einmal für die ganze Aufnahme auf Xsens
  gedreht und verschoben wurde (ohne Skalierung). Enthält Tiefen- und Laufwegfehler.
- PA-MPJPE: pro Frame ausgerichtet, misst nur die Haltung.
- Winkel: RMSE und mittlere Abweichung. GemX und Xsens setzen manche Gelenke an leicht
  andere Stellen (z. B. Schulter), das zeigt sich als gleichbleibende Abweichung.
- Tiefe: wie viel größer der Ortsfehler entlang der Blickrichtung der Kamera ist als quer dazu.
  Im Gelenkverlauf zeigt eine gelbe Linie den Tiefenanteil pro Frame.
- Bland-Altman (Knopf im Gelenkverlauf): Differenz GemX - Xsens je Frame über dem mittleren
  Winkel, mit Bias, 95-%-Grenzen und Trend (wächst der Fehler mit dem Winkel?).
- Die Auswertung nennt außerdem Median/95. Perzentil, Drift, den Haltungsfehler nach Abzug der
  festen Gelenkversätze und je Winkel Bewegungsumfang und größte Beugung.

Updates: Gibt es eine neue Version, erscheint oben in der Übersicht ein Update-Knopf (das
Programm fragt dafür beim Start GitHub nach dem neuesten Release, sonst nichts). Installieren
ersetzt nur die Programmdateien und startet neu; die alten liegen danach in
%LOCALAPPDATA%\\GemX-Vergleich\\update-backup.

Erster Start dauert etwas länger (Modell wird vorbereitet). Nur Windows 64 bit.
Enthaltene Software: Python (PSF-Lizenz), PyTorch (BSD), NVIDIA SOMA-X und Warp (Apache 2.0),
Modelldaten in app/assets/soma (Apache 2.0, siehe LICENSE dort).
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    py = OUT / "python"
    shutil.copytree(PY, py, ignore=shutil.ignore_patterns(
        "Doc", "Scripts", "Tools", "include", "libs", "share", "tcl", "test", "idlelib",
        "tkinter", "turtledemo", "site-packages", "__pycache__", "*.txt"))
    shutil.copytree(SITE, py / "Lib" / "site-packages", ignore=SKIP_SITE)
    sys32 = Path(r"C:\Windows\System32")
    for dll in VC_DLLS:
        if (sys32 / dll).exists():
            shutil.copy2(sys32 / dll, py / "Lib" / "site-packages" / "torch" / "lib")
    shutil.copy2(PY / "LICENSE.txt", py / "LICENSE.txt")
    app = OUT / "app"
    app.mkdir()
    for f in APP_FILES:
        shutil.copy2(HERE / f, app)
    shutil.copytree(HERE / "assets" / "soma", app / "assets" / "soma")
    ver = datetime.datetime.now().strftime("%Y.%m.%d.%H%M")
    (app / "version.json").write_text(json.dumps({"version": ver, "runtime": runtime()}), encoding="utf-8")
    (OUT / "GemX-Vergleich starten.bat").write_text(START, encoding="ascii")
    (OUT / "LIESMICH.txt").write_text(README, encoding="utf-8")
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(f"{OUT}: {len(files)} Dateien, {sum(p.stat().st_size for p in files) / 1e9:.2f} GB")
    if "--zip" in sys.argv:
        zip_path = Path(shutil.make_archive(str(OUT), "zip", OUT.parent, OUT.name))
        print(f"{zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")
    iscc = next((p for p in ISCC if p.exists()), None)
    if not iscc:
        print("Inno Setup fehlt (winget install JRSoftware.InnoSetup), kein Installer gebaut.")
        return
    subprocess.run([str(iscc), "/Q", f"/DAppVer={ver}", str(HERE / "installer.iss")], check=True)
    setup = HERE / "dist" / "GemX-Vergleich-Setup.exe"
    print(f"{setup} ({setup.stat().st_size / 1e9:.2f} GB)")
    if "--release" in sys.argv:
        i = sys.argv.index("--release")
        notes = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else ""
        release(ver, app, setup, notes)


def runtime():
    """Fingerprint of the bundled Python packages: an update with another fingerprint needs its setup."""
    names = sorted(p.name for p in SITE.glob("*.dist-info"))
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


def update_package(ver, app):
    """dist/GemX-Vergleich-App-<ver>.zip (the app files and version.json) and dist/update.json."""
    import zipfile
    zp = HERE / "dist" / f"GemX-Vergleich-App-{ver}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in APP_FILES + ["version.json"]:
            z.write(app / f, f)
    info = {"version": ver, "runtime": runtime(), "zip": zp.name,
            "sha256": hashlib.sha256(zp.read_bytes()).hexdigest()}
    up = HERE / "dist" / "update.json"
    up.write_text(json.dumps(info), encoding="utf-8")
    return zp, up


def release(ver, app, setup, notes):
    zp, up = update_package(ver, app)
    subprocess.run(["gh", "release", "create", f"v{ver}", str(setup), str(zp), str(up), "--repo", REPO,
                    "--title", f"GemX-Vergleich {ver}", "--notes", notes or f"Version {ver}"], check=True)
    print(f"Release v{ver} veröffentlicht: https://github.com/{REPO}/releases/tag/v{ver}")


if __name__ == "__main__":
    main()
