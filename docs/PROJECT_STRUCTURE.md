# TicketNow — Project Structure

## 1. Architecture Overview

TicketNow is an AI-powered customer support ticket classification system. It follows a
clean **frontend / backend / algorithm** separation:

- A **React (Vite) frontend** provides the user-facing chatbot and an admin dashboard.
- A **FastAPI backend** exposes REST endpoints, manages the SQLite database, and serves
  predictions at request time.
- An **algorithm layer** (offline, notebook-style scripts) handles EDA and model
  training, producing the artifacts (trained models, encoders, vectorizers) that the
  backend loads at runtime.

This is a standard three-tier ML product structure: an offline training tier that
produces model artifacts, an online serving tier (backend) that consumes them, and a
presentation tier (frontend) that end users interact with.

## 2. What Each Folder Does

| Folder | Purpose |
|---|---|
| `frontend/` | React + Vite single-page app: user chatbot UI and admin dashboard/login. Untouched — Vite requires `src/`, `public/`, `index.html` in fixed locations. |
| `backend/` | FastAPI application: API routes, SQLite access, and the ML inference wrapper (`ml_engine.py`) that loads trained models and serves predictions. |
| `algorithm/` | AI/ML layer — EDA (`eda_analysis.py`), model training (`model_training.py`), datasets (`.csv`), generated plots (`eda_plots/`), and trained model artifacts (`trained_models/`). Formerly named `EDA AND MODEL`; renamed only for consistent, professional naming. |
| `docs/` | Project documentation, including this file. |

No `database/`, `integrations/`, `shared/`, or `config/` folders were created because
this project doesn't have separate schema/migration files, third-party API
integrations, cross-cutting shared utilities, or a standalone config layer — the
SQLite database lives with the backend service that owns it (`backend/ticketnow.db`),
and Vite's own `vite.config.js` stays with the frontend, per framework convention.

## 3. File → Category Table

| File/Folder | Category | Notes |
|---|---|---|
| `frontend/src/App.jsx`, `main.jsx`, `App.css`, `index.css` | Frontend | App shell and global styles |
| `frontend/src/pages/UserChatbot.jsx` | Frontend | End-user chat UI |
| `frontend/src/pages/AdminDashboard.jsx` | Frontend | Admin analytics/dashboard UI |
| `frontend/src/pages/AdminLogin.jsx` | Frontend | Admin authentication UI |
| `frontend/public/`, `vite.config.js`, `index.html`, `package.json` | Frontend (config) | Vite/React project scaffolding, fixed by framework |
| `backend/main.py` | Backend | FastAPI app, routes, DB init, CORS |
| `backend/ml_engine.py` | Backend (ML inference) | Loads trained models and serves predictions |
| `backend/ticketnow.db` | Backend (database) | SQLite database file, owned by the backend service |
| `backend/test_*.py`, `debug_risk.py`, `verify_fix.py` | Backend (dev/debug scripts) | Ad-hoc scripts used during development |
| `algorithm/eda_analysis.py` | Algorithm | Exploratory data analysis |
| `algorithm/model_training.py` | Algorithm | Trains and exports the ML models |
| `algorithm/*.csv` | Algorithm (data) | Raw and processed datasets |
| `algorithm/eda_config.json` | Algorithm (config) | EDA parameters |
| `algorithm/eda_plots/` | Algorithm (output) | Generated charts from EDA |
| `algorithm/trained_models/` | Algorithm (output) | Serialized models consumed by the backend |
| `README.md` | Docs | Project overview |
| `docs/PROJECT_STRUCTURE.md` | Docs | This document |

## 4. Data Flow

```
User → Frontend (React chatbot) → Backend API (FastAPI)
     → Algorithm layer's trained models (loaded via ml_engine.py)
     → Prediction (category / intent / priority / risk)
     → Backend → Response → Frontend (displayed to user / admin dashboard)
```

The `algorithm/` layer is offline: `model_training.py` is run ahead of time to produce
the artifacts in `algorithm/trained_models/`. At request time, the backend never
re-trains anything — it only loads those artifacts through `ml_engine.py` and runs
inference.

## 5. Five-Minute Presentation Summary

1. **What it is**: TicketNow automatically classifies incoming customer support
   tickets — predicting category, intent, priority, and risk — so tickets can be
   routed to the right team without manual triage.
2. **Frontend**: A React chatbot lets users submit tickets conversationally; an admin
   dashboard shows classified tickets and analytics.
3. **Backend**: A FastAPI service receives ticket text, stores it in SQLite, and calls
   the ML engine for real-time predictions.
4. **Algorithm**: XGBoost models (plus a TF-IDF vectorizer and label encoders) were
   trained offline on a labeled support-ticket dataset, after an EDA phase that shaped
   feature engineering decisions.
5. **Why this structure**: Separating `frontend/`, `backend/`, and `algorithm/` mirrors
   how the system actually runs in production — a served UI, a serving API, and an
   offline training pipeline — making it easy for a reviewer to see where each
   responsibility lives.

## 6. What Changed vs. the Original Upload

- Renamed `EDA AND MODEL/` → `algorithm/` (folder rename only).
- Updated **one line** in `backend/main.py` (`MODEL_DIR` path) to point at the renamed
  folder — this is the only code change made, and it's a path fix required by the
  rename, not a logic change.
- Added this `docs/` folder.
- Excluded from the delivered structure (not part of the source code, and shouldn't
  ship with it):
  - `misrouting/` — a full Python virtual environment (hundreds of MB of installed
    packages like matplotlib, wordcloud, etc.) that was bundled into the original ZIP.
    Recreate it locally with your usual `venv` + `pip install -r requirements.txt`
    workflow.
  - `frontend/node_modules/`, `frontend/dist/` — reinstall with `npm install`, rebuild
    with `npm run build`.
  - `backend/__pycache__/`, `.git/` — caches/VCS metadata, not source.

No other files were moved, renamed, or edited. All application logic, API contracts,
database schema, and UI code are byte-for-byte identical to the uploaded version.
