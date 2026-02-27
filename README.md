# MonkeyCamSP - Zuweisungstool

Ein Tool zur automatischen Verarbeitung von Überwachungsvideos (Türkamera/MonkeyCam) und der anschließenden Zuweisung von erkannten Personen/Aktionen ("Kommt", "Geht", "Fehler").

Das Programm ist plattformunabhängig und funktioniert einwandfrei unter Windows, macOS und Linux.

## Voraussetzungen (Windows & Linux)

1. **Python installieren:** Stelle sicher, dass Python (Version 3.12 oder neuer) installiert ist. (Download unter: [python.org](https://www.python.org/downloads/))
2. **uv installieren (Empfohlen):** Dieses Projekt nutzt den superschnellen Python-Paketmanager `uv`.
   - **Windows:** Öffne die PowerShell und führe folgenden Befehl aus:
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - **Linux/macOS:** Öffne ein Terminal und führe folgenden Befehl aus:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

## Installation

1. Lade diesen Projektordner herunter und entpacke ihn.
2. Öffne ein Terminal (oder die Windows-Eingabeaufforderung / PowerShell) in diesem Ordner.
3. Installiere alle Abhängigkeiten mit einem einzigen Befehl:
   ```bash
   uv sync
   ```

## Programm starten

Du kannst das Programm nach der Installation jederzeit mit diesem einfachen Befehl im Projektordner starten:

```bash
uv run monkeycamsp
```

Dadurch wird das Tool automatisch konfiguriert und in deinem Standard-Webbrowser geöffnet. 
*(Alternativ funktioniert auch: `uv run python app.py`)*

## Kurzanleitung zur Bedienung

1. **Videos Verarbeiten (Schritt 1):** 
   - Klicke auf "Ordner wählen" und wähle den Ordner aus, der deine `.mp4` Videos enthält.
   - Klicke auf "Videos verarbeiten".
   - Es öffnen sich nacheinander zwei Fenster, die dir direkt im Videobild zeigen, was du tun musst:
     1. Markiere im ersten Fenster den **Zeitstempel** und drücke `ENTER` oder `SPACE`.
     2. Markiere im zweiten Fenster die **Tür** und drücke erneut `ENTER`.
   - Warte, bis der Ladebalken 100% erreicht hat, und klicke auf "Weiter zur Zuweisung".

2. **Bilder Zuweisen / Triage (Schritt 2):**
   - Im Zuweisungsfenster siehst du links das aktuelle Bild und rechts die Optionen.
   - Wähle die erkannten Personen aus. (Fehlt ein Name? Tippe ihn einfach in das Feld ein und bestätige mit `ENTER` – er wird für die Zukunft gespeichert!)
   - Klicke auf die entsprechende Aktion (`Kommt`, `Geht` oder `Fehler`), um die Zuordnung zu speichern und direkt zum nächsten Bild zu springen.
   - **Hast du dich verklickt?** Nutze einfach den `Zurück (Rückgängig)` Button ganz unten.
   
3. **Ergebnisse speichern:**
   - Wenn alle Bilder bearbeitet sind, wird dir eine Erfolgsmeldung angezeigt.
   - Die Ergebnisse werden automatisch als `ergebnisse.csv` Datei im Ordner `camsp` gespeichert und können in Excel oder anderen Tabellenkalkulationen weiterverwendet werden.