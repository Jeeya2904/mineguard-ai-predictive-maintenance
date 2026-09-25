import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURES=["temperature","vibration","pressure","rpm","operating_hours","load_percentage"]

def make_dataset(n=3000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "temperature":rng.normal(70,13,n).clip(25,115),
        "vibration":rng.normal(5.2,2,n).clip(.2,15),
        "pressure":rng.normal(100,14,n).clip(55,150),
        "rpm":rng.normal(1450,220,n).clip(500,2200),
        "operating_hours":rng.uniform(100,10000,n),
        "load_percentage":rng.normal(65,18,n).clip(10,100)
    })
    score=((df.temperature>88).astype(int)+(df.vibration>7.5).astype(int)+
           (df.pressure>125).astype(int)+(df.rpm>1850).astype(int)+
           (df.operating_hours>8000).astype(int)+(df.load_percentage>88).astype(int))
    df["failure"]=(score>=2).astype(int)
    return df

def train_and_save():
    os.makedirs("model",exist_ok=True); os.makedirs("data",exist_ok=True)
    df=make_dataset(); df.to_csv("data/sensor_data.csv",index=False)
    xtr,xte,ytr,yte=train_test_split(df[FEATURES],df.failure,test_size=.2,random_state=42,stratify=df.failure)
    model=RandomForestClassifier(n_estimators=200,max_depth=12,random_state=42,class_weight="balanced")
    model.fit(xtr,ytr)
    print(f"Accuracy: {accuracy_score(yte,model.predict(xte)):.3f}")
    print(f"ROC-AUC: {roc_auc_score(yte,model.predict_proba(xte)[:,1]):.3f}")
    joblib.dump(model,"model/mineguard_rf.joblib")
    return model

if __name__=="__main__": train_and_save()
