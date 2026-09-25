# MineGuard AI — Predictive Maintenance

End-to-end predictive-maintenance prototype for mining equipment. Six sensor readings are processed by a Random Forest classifier and exposed through a Flask REST API and browser dashboard.

## Features
- Temperature, vibration, pressure, humidity, gas and sound inputs
- Random Forest failure classifier
- Synthetic training-data generator
- REST API: `/api/predict`, `/api/history`, `/api/metrics`, `/api/health`
- Interactive dashboard
- Docker deployment

## Run locally
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
python model/train_model.py
python app.py
```
Open `http://localhost:5000`.

## API
POST `/api/predict` with JSON such as `{"temperature":72,"vibration":5,"pressure":101,"humidity":55,"gas_level":35,"sound_level":68}`.

## Architecture
Sensors → feature inputs → Random Forest → failure probability → Flask API → dashboard.

This is a prototype using synthetic sensor data, not a certified industrial safety system.