; Installer für GemX-Vergleich (Inno Setup 6). Wird von build_bundle.py aufgerufen:
;   ISCC /DAppVer=JJJJ.MM.TT installer.iss   -> dist\GemX-Vergleich-Setup.exe
; Quelle ist der fertige Ordner dist\GemX-Vergleich. Installiert ohne Adminrechte in den
; Benutzerordner, startet ohne Konsolenfenster (pythonw), legt Startmenü- und auf Wunsch
; Desktop-Verknüpfung mit gemx.ico an.

#ifndef AppVer
  #define AppVer "1.0"
#endif

[Setup]
AppId=GemX-Vergleich
AppName=GemX-Vergleich
AppVersion={#AppVer}
AppPublisher=Predimo
DefaultDirName={localappdata}\Programs\GemX-Vergleich
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=GemX-Vergleich-Setup
SetupIconFile=gemx.ico
UninstallDisplayIcon={app}\app\gemx.ico
UninstallDisplayName=GemX-Vergleich
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "de"; MessagesFile: "compiler:Languages\German.isl"

[Tasks]
Name: "desktopicon"; Description: "Verknüpfung auf dem Desktop anlegen"

[Files]
Source: "dist\GemX-Vergleich\*"; DestDir: "{app}"; Excludes: "GemX-Vergleich starten.bat"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\GemX-Vergleich"; Filename: "{app}\python\pythonw.exe"; Parameters: "-E -s ""{app}\app\vergleich_app.py"""; WorkingDir: "{app}"; IconFilename: "{app}\app\gemx.ico"; Comment: "Xsens und GemX vergleichen"
Name: "{autodesktop}\GemX-Vergleich"; Filename: "{app}\python\pythonw.exe"; Parameters: "-E -s ""{app}\app\vergleich_app.py"""; WorkingDir: "{app}"; IconFilename: "{app}\app\gemx.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-E -s ""{app}\app\vergleich_app.py"""; WorkingDir: "{app}"; Description: "GemX-Vergleich jetzt starten"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
; Python legt beim Laufen __pycache__-Ordner an
Type: filesandordirs; Name: "{app}"
