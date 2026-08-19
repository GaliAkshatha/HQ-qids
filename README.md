# QIDS — Quantum-Assisted Intelligent Detection & Defense System

> A quantum-assisted intrusion detection and defense platform combining intelligent security experiments, classical machine learning, quantum verification, risk assessment, automated defense, and incident management.

## Overview

QIDS is an end-to-end security system designed to demonstrate how suspicious activity can be detected, verified, assessed, and handled through a unified security pipeline.

Instead of treating intrusion detection as only a classification problem, QIDS connects detection with downstream risk assessment, defense, and incident management.

```text
Security Activity
       ↓
     Agent
       ↓
    Detection
       ↓
Quantum Verification
       ↓
 Risk Assessment
       ↓
 Automated Defense
       ↓
    Incident
```

The system provides a web dashboard for running controlled security experiments and observing the resulting security pipeline in real time.

---

## Key Capabilities

### 1 Agent-Driven Security Experiments

QIDS provides controlled security agents that generate normal and adversarial scenarios for experimentation.

Experiments can be configured by:

- Security scenario
- Number of sessions
- Normal / adversarial / mixed mode
- Quantum routing mode

The generated activity is passed through the same detection and response pipeline used by the rest of the system.

### 2 Classical Intrusion Detection

Classical machine-learning models form the primary detection layer.

The project includes models such as:

- Logistic Regression
- Random Forest
- XGBoost-based detection components

### 3 Quantum-Assisted Verification

Suspicious activity can be routed to quantum machine-learning models for additional verification.

The quantum layer includes:

- Variational Quantum Classifier (VQC)
- Quantum Support Vector Machine (QSVM)
- Qiskit
- Qiskit Machine Learning
- Qiskit Aer

The system supports quantum/classical routing rather than treating the quantum models as isolated experiments.

### 4 Risk Assessment

Detection and verification results are passed into the risk layer, which evaluates the resulting security event and produces the information required by the downstream defense pipeline.

### 5 Defense Engine

The defense stage processes the security decision and executes the corresponding defense action.

Defense actions are recorded as part of the security event lifecycle so that the response can be inspected afterward.

### 6 Incident Management

Security events are represented as incidents with a traceable event lifecycle:

```text
DETECTION_CREATED
        ↓
QUANTUM_ROUTING_REQUESTED
        ↓
QUANTUM_VERIFICATION_COMPLETED
        ↓
HYBRID_DECISION_CREATED
        ↓
RISK_ASSESSED
        ↓
DEFENSE_ACTION_EXECUTED
        ↓
INCIDENT_RESOLVED / INCIDENT_ESCALATED
```

Each stage can contain the associated reason codes and evidence returned by the backend.

### 7 Real-Time Event Streaming

The dashboard uses Server-Sent Events (SSE) to display live security activity.

The pipeline visualization reflects actual backend events:

```text
Agent → Detection → Quantum → Risk → Defense → Incident
```

---

## Dashboard

QIDS includes five main dashboard views.

| Page | Description |
|---|---|
| **Dashboard** | System readiness, pipeline activity, aggregate metrics, and live event stream |
| **Experiments** | Configure and execute controlled security experiments |
| **Incidents** | Inspect incident state, evidence, and event timelines |
| **Agents** | View agent sessions, scenarios, and normal/adversarial activity |
| **Models** | View classical and quantum model comparison results |

---

## System Architecture

```text
                         ┌──────────────────────┐
                         │     QIDS Dashboard   │
                         │  React + TypeScript  │
                         └──────────┬───────────┘
                                    │
                               REST + SSE
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       QIDS API       │
                         │   Flask + Gunicorn   │
                         └──────────┬───────────┘
                                    │
                              Redis Streams
                                    │
          ┌─────────────┬───────────┼───────────┬─────────────┐
          ▼             ▼           ▼           ▼             ▼
     Detection      Quantum       Risk       Defense       Incident
      Worker        Worker       Worker       Worker        Worker
          │             │           │           │             │
          ▼             ▼           ▼           ▼             ▼
     Classical      VQC/QSVM    Risk Engine  Defense       Incident
      Models        + Aer                     Engine        Manager
```

Redis provides the runtime communication layer between the distributed workers.

---

## Research & Model Comparison

The intrusion-detection experiments use the **NSL-KDD dataset** together with controlled experiment data generated by the project's agent environment.

The current model comparison includes:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 81.3% | 100% | 66.7% | 80.0% |
| Random Forest | 100% | 100% | 100% | 100% |
| QSVM | 100% | 100% | 100% | 100% |
| VQC | 93.8% | 100% | 88.9% | 94.1% |

These values represent the project's controlled experimental comparison and should be interpreted in that context rather than as production IDS benchmarks.

---

## Technology Stack

### AI / Machine Learning

- Python
- scikit-learn
- XGBoost
- NumPy
- SciPy
- Joblib

### Quantum Computing

- Qiskit
- Qiskit Machine Learning
- Qiskit Aer
- VQC
- QSVM

### Backend

- Flask
- Gunicorn
- Redis
- Redis Streams
- REST API
- Server-Sent Events (SSE)

### Frontend

- React
- TypeScript
- Vite
- CSS

---

## Repository Structure

```text
.
├── Data/                 # Dataset and processed data
├── artifacts/            # Trained ML and quantum artifacts
├── config/               # Runtime configuration
├── docker/               # Docker-related configuration
├── docs/                 # Project documentation
├── experiments/          # Experiment and research scripts
├── frontend/             # React + Vite dashboard
├── logs/                 # Runtime logs
├── reports/              # Evaluation and experiment reports
├── results/              # Generated results and visualizations
├── src/
│   ├── api/              # Flask API
│   ├── detection/        # Detection components
│   ├── defense/          # Defense engine
│   ├── hybrid/           # Hybrid decision logic
│   ├── incident/         # Incident management
│   ├── quantum/          # VQC / QSVM components
│   ├── routing/          # Quantum/classical routing
│   └── runtime/          # Workers and runtime services
├── tests/                # Automated tests
├── requirements.txt
├── run.py
├── wsgi.py
├── Procfile
├── docker-compose.yml
├── DEPLOYMENT.md
└── .env.example
```

---

## Running Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm
- Redis
- Approximately 2 GB free disk space for the ML and quantum dependencies

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Start the Complete System

```bash
python run.py
```

Then open:

```text
http://localhost:5173
```

The launcher starts the local runtime components required by the system.

### Backend Health

```bash
curl http://localhost:8080/api/health
curl http://localhost:8080/api/ready
```

A ready system reports:

```json
{
  "status": "ready",
  "checks": {
    "api": true,
    "redis": true,
    "classical_detector": true,
    "vqc": true,
    "qsvm": true,
    "application_detector": true
  }
}
```

---

## Running an Experiment

1. Open the **Experiments** page.
2. Select a scenario such as `neptune_flood`.
3. Select a small number of sessions.
4. Choose `normal`, `adversarial`, or `mixed`.
5. Set Quantum to `Auto`.
6. Click **Start Session**.
7. Open the Dashboard to observe the live event stream.
8. Open **Incidents** to inspect the resulting timeline.
9. Open **Agents** and **Models** to inspect the corresponding activity and results.

---

<p align="center">
  Built by <a href="https://github.com/GaliAkshatha">Akshatha</a>
</p>

