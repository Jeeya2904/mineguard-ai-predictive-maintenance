from app import app

def test_health():
    client=app.test_client()
    response=client.get('/api/health')
    assert response.status_code==200

def test_prediction():
    client=app.test_client()
    payload={'temperature':72,'vibration':5,'pressure':101,'rpm':1450,'operating_hours':3500,'load_percentage':65}
    response=client.post('/api/predict',json=payload)
    assert response.status_code==200
    data=response.get_json()
    assert data['risk'] in ['LOW','MEDIUM','HIGH']
    assert 0<=data['failure_probability']<=1
    assert data['maintenance_recommendation']
