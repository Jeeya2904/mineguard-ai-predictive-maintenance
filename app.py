from flask import Flask, render_template, request, jsonify
import os, json, joblib
from datetime import datetime, timezone

app=Flask(__name__)
MODEL_PATH="model/mineguard_rf.joblib"
HISTORY_PATH="data/prediction_history.json"
FEATURES=["temperature","vibration","pressure","rpm","operating_hours","load_percentage"]

def load_model():
    if not os.path.exists(MODEL_PATH):
        from model.train_model import train_and_save
        train_and_save()
    return joblib.load(MODEL_PATH)

def recommendation(risk):
    return {"HIGH":"Stop/inspect equipment and schedule immediate maintenance.",
            "MEDIUM":"Schedule inspection soon and monitor sensor trends.",
            "LOW":"Continue operation with routine monitoring."}[risk]

@app.get("/")
def index(): return render_template("index.html")

@app.get("/api/health")
def health(): return jsonify({"status":"healthy","model":os.path.exists(MODEL_PATH)})

@app.post("/api/predict")
def predict():
    try:
        payload=request.get_json(force=True)
        values=[float(payload[k]) for k in FEATURES]
        model=load_model()
        probability=float(model.predict_proba([values])[0][1])
        risk="HIGH" if probability>=.70 else ("MEDIUM" if probability>=.40 else "LOW")
        record={"timestamp":datetime.now(timezone.utc).isoformat(),"risk":risk,
                "failure":int(probability>=.50),"failure_probability":round(probability,4),
                "maintenance_recommendation":recommendation(risk),**dict(zip(FEATURES,values))}
        os.makedirs("data",exist_ok=True)
        history=[]
        if os.path.exists(HISTORY_PATH):
            try:
                with open(HISTORY_PATH) as f: history=json.load(f)
            except Exception: pass
        history.append(record)
        with open(HISTORY_PATH,"w") as f: json.dump(history[-100:],f,indent=2)
        return jsonify(record)
    except (KeyError,TypeError,ValueError) as e:
        return jsonify({"error":f"Invalid sensor input: {e}"}),400

@app.get("/api/history")
def history():
    if not os.path.exists(HISTORY_PATH): return jsonify([])
    with open(HISTORY_PATH) as f: return jsonify(json.load(f))

@app.get("/api/metrics")
def metrics():
    model=load_model()
    return jsonify({"model":"Random Forest Classifier","features":FEATURES,
                    "estimators":model.n_estimators,"status":"ready"})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",5000)),debug=False)
