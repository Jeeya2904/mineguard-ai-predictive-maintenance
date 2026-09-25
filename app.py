from flask import Flask, render_template, request, jsonify
import os, json, joblib, numpy as np
from datetime import datetime

app = Flask(__name__)
MODEL_PATH = os.path.join("model", "mineguard_rf.joblib")
HISTORY_PATH = os.path.join("data", "prediction_history.json")

FEATURES = ["temperature", "vibration", "pressure", "humidity", "gas_level", "sound_level"]

def load_model():
    if not os.path.exists(MODEL_PATH):
        from model.train_model import train_and_save
        train_and_save()
    return joblib.load(MODEL_PATH)

def save_prediction(record):
    os.makedirs("data", exist_ok=True)
    history=[]
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as f: history=json.load(f)
        except Exception: history=[]
    history.append(record)
    with open(HISTORY_PATH,"w") as f: json.dump(history[-100:], f, indent=2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({"status":"healthy","model":os.path.exists(MODEL_PATH)})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload=request.get_json(force=True)
        values=[float(payload[k]) for k in FEATURES]
        model=load_model()
        pred=int(model.predict([values])[0])
        probability=float(model.predict_proba([values])[0][1])
        risk="HIGH" if probability >= .70 else ("MEDIUM" if probability >= .40 else "LOW")
        record={"timestamp":datetime.utcnow().isoformat()+"Z","risk":risk,"failure":pred,
                "probability":round(probability,4),**dict(zip(FEATURES,values))}
        save_prediction(record)
        return jsonify(record)
    except Exception as e:
        return jsonify({"error":str(e)}),400

@app.route("/api/history")
def history():
    if not os.path.exists(HISTORY_PATH): return jsonify([])
    with open(HISTORY_PATH) as f: return jsonify(json.load(f))

@app.route("/api/metrics")
def metrics():
    model=load_model()
    return jsonify({"model":"Random Forest Classifier","features":FEATURES,
                    "estimators":model.n_estimators,"status":"ready"})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",5000)),debug=False)
