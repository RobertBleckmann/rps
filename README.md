# RPS Vision

Ein Computer-Vision-Projekt für **Schere-Stein-Papier** mit Webcam, MediaPipe Hand-Tracking und einem trainierbaren Klassifikationsmodell.

## Features

- Hand-Landmark-Erkennung mit MediaPipe (`models/hand_landmarker.task`)
- Datensammlung für Gesten (`r`, `p`, `s`)
- Training eines Klassifikators (scikit-learn Logistic Regression)
- Live-Inferenz mit Glättung
- Spielmodus gegen CPU mit UI, Countdown und Score

## Wesentliche Technologien

- Python
- OpenCV
- MediaPipe
- LogisticRegression (scikit-learn)

## Voraussetzungen

- Python 3.12 (empfohlen)
- Webcam
- Windows (aktueller Code nutzt `cv2.CAP_DSHOW`)

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Projektstruktur

```text
rps-vision/
  src/rps/
    capture.py     # Datensammlung
    train.py       # Modelltraining
    infer.py       # Live-Klassifikation
    game.py        # Schere-Stein-Papier-Spiel
  models/
    hand_landmarker.task
  data/
    samples.csv
    models/rps_model.joblib
```

## Schnellstart

### 1) Samples aufnehmen (optional, für eigenes Training)

```powershell
python -m src.rps.capture
```

Steuerung im Aufnahmefenster:
- `r` = Rock
- `p` = Paper
- `s` = Scissors
- `ESC` = Beenden

Die Daten werden nach `data/samples.csv` geschrieben.

### 2) Modell trainieren

```powershell
python -m src.rps.train
```

Output:
- Metriken (Confusion Matrix, Classification Report)
- Modell unter `data/models/rps_model.joblib`

### 3) Live-Inferenz starten

```powershell
python -m src.rps.infer
```

### 4) Spielmodus starten

```powershell
python -m src.rps.game
```

Steürung im Spiel:
- `Space` oder Klick auf **Start**: Runde starten
- `R` oder Klick auf **Reset**: Score zurücksetzen
- `ESC`: Beenden

## Docker

Image bauen:

```powershell
docker build -t rps-vision .
```

Container starten (nutzt per Default `python -m src.rps.game`):

```powershell
docker run --rm -it rps-vision
```

Hinweis: Für Webcam-/GUI-Nutzung sind je nach Host-OS zusätzliche Docker-Optionen erforderlich.

## Troubleshooting

- **`Could not open webcam`**: Kamera ist belegt oder Berechtigung fehlt.
- **`Model not found: .../data/models/rps_model.joblib`**: zuerst `python -m src.rps.train` ausführen.
- **MediaPipe-Modell fehlt**: `models/hand_landmarker.task` muss vorhanden sein.


