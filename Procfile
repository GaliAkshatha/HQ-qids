web: gunicorn wsgi:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
detection-worker: python -m src.runtime.services.detection_worker
quantum-worker: python -m src.runtime.services.quantum_worker
risk-worker: python -m src.runtime.services.risk_worker
defense-worker: python -m src.runtime.services.defense_worker
incident-worker: python -m src.runtime.services.incident_worker
