import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
FEATURES=["temperature","vibration","pressure","humidity","gas_level","sound_level"]
def make_dataset(n=2500,seed=42):
 rng=np.random.default_rng(seed)
 df=pd.DataFrame({"temperature":rng.normal(65,12,n).clip(20,110),"vibration":rng.normal(5,2,n).clip(.2,15),"pressure":rng.normal(100,12,n).clip(55,145),"humidity":rng.normal(55,15,n).clip(10,95),"gas_level":rng.normal(35,18,n).clip(0,100),"sound_level":rng.normal(68,12,n).clip(30,120)})
 risk=(df.temperature>82).astype(int)+(df.vibration>7).astype(int)+(df.pressure>120).astype(int)+(df.gas_level>65).astype(int)+(df.sound_level>88).astype(int)+(df.humidity>82).astype(int)
 df["failure"]=(risk>=2).astype(int); return df
def train_and_save():
 os.makedirs("model",exist_ok=True); os.makedirs("data",exist_ok=True); df=make_dataset(); df.to_csv("data/sensor_data.csv",index=False)
 Xtr,Xte,ytr,yte=train_test_split(df[FEATURES],df.failure,test_size=.2,random_state=42,stratify=df.failure)
 model=RandomForestClassifier(n_estimators=150,max_depth=10,random_state=42,class_weight="balanced"); model.fit(Xtr,ytr)
 print("Accuracy:",accuracy_score(yte,model.predict(Xte)),"ROC-AUC:",roc_auc_score(yte,model.predict_proba(Xte)[:,1]))
 joblib.dump(model,"model/mineguard_rf.joblib"); return model
if __name__=="__main__": train_and_save()
