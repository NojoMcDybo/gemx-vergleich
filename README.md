# GemX-Vergleich

Vergleicht eine Xsens-Aufnahme (MVNX) mit dem GemX-Ergebnis (`hpe_results.pt`) derselben Bewegung:
Skelett über dem Video, Haltungs- und Ortsfehler pro Frame, Gelenkwinkel mit Bland-Altman-Auswertung,
Tiefenfehler, Seitenvertauschungen. Läuft lokal unter Windows; die Ansicht öffnet sich als eigenes Fenster.

**Installieren:** `GemX-Vergleich-Setup.exe` aus dem [neuesten Release](../../releases/latest) ausführen
(keine Adminrechte nötig). Neue Versionen bietet das Programm danach selbst an (Knopf oben in der Übersicht).

## Dateien

| Datei | Aufgabe |
|---|---|
| `vergleich_app.py` | lokaler Server (127.0.0.1), Bibliothek, Vergleich, Teilen, Updates |
| `compare.py` | Kennzahlen (MPJPE, PA-MPJPE, Winkel, Tiefe, Bland-Altman, Versatzbereinigung) und Tabellen |
| `compare_view.html` | die Oberfläche (eine Seite, nutzt `nojo-ui.css/js`) |
| `gemx_joints.py` | Gelenkpositionen aus den SOMA-Parametern von GemX |
| `clapsync.py` | MVNX lesen, Video und Xsens über Klatscher synchronisieren |
| `build_bundle.py`, `installer.iss` | Setup bauen und als Release veröffentlichen |

## Entwickeln

Python 3.11 mit einer venv `.venv-soma` (torch CPU, numpy, py-soma-x 0.3.3); die SOMA-Modelldaten
liegen in `assets/soma` und sind nicht im Repository.

```
.venv-soma\Scripts\python vergleich_app.py            # starten
.venv-soma\Scripts\python test_compare.py             # Tests
.venv-soma\Scripts\python test_vergleich_app.py
.venv-soma\Scripts\python build_bundle.py --release "Was neu ist"   # Setup + Update veröffentlichen
```

Ein Release enthält das Setup, ein kleines Paket nur mit den Programmdateien und `update.json`
(Version, SHA-256, Fingerabdruck der Python-Pakete). Installierte Programme laden nur das kleine
Paket; ändern sich die Python-Pakete, verweisen sie auf das neue Setup.
