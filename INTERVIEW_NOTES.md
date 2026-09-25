# Interview Notes

## What does the project do?
MineGuard AI predicts mining-equipment failure risk from six operating signals: temperature, vibration, pressure, RPM, operating hours, and load percentage.

## Why Random Forest?
It works well for tabular sensor data, captures nonlinear relationships, is relatively easy to train, and provides probability estimates and feature importance.

## What is the output?
The API returns failure probability, LOW/MEDIUM/HIGH risk, and a maintenance recommendation.

## How is the dataset handled?
A reproducible synthetic dataset is generated with NumPy using seed 42. In a production system, validated historical sensor and maintenance records would replace it.

## How would you improve it?
Add real sensor ingestion, time-series features, drift monitoring, model versioning, authentication, cloud deployment, alerting, and domain validation with maintenance engineers.