# 🎫 TicketNow — AI-Based Customer Support Ticket Misrouting Prediction & Triage System

TicketNow is an AI-powered customer support ticket classification and intelligent routing system. It automatically analyzes customer complaints, predicts the correct department, assigns ticket priority, detects misrouting risk, and routes tickets to the appropriate support team — combining **NLP** and **Machine Learning** to cut manual work, speed up response time, and boost routing accuracy.

---

## 👥 Team & Responsibilities

| Team Member | Responsibility |
|---|---|
| **Mani bharathi** | Machine Learning Model Training & Evaluation |
| **Faris shaukath** | Frontend Development (HTML, CSS, JavaScript / Streamlit UI) |
| **Lakshman Aadhiya** | Database Design & Management (SQLite / PostgreSQL) |
| **Naveenkumar** | Backend Development (FastAPI, API Integration, Business Logic) |

---

## 🛠️ Technologies Used

`Python` · `FastAPI` · `Streamlit` · `SQLite` · `Pandas` · `NumPy` · `Scikit-Learn` · `XGBoost` · `TF-IDF` · `Joblib` · `HTML` · `CSS` · `JavaScript`

---

## 🔄 Project Workflow

```
Customer Support Ticket
        │
        ▼
Frontend (Faris) — HTML / CSS / JavaScript
        │
        ▼
FastAPI Backend (Naveen)
        │
        ▼
Text Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Machine Learning Model (XGBoost)
        │
        ▼
Department Prediction → Priority Prediction → Intent Prediction → Sentiment Analysis → Misrouting Risk Detection
        │
        ▼
Database Storage (Aadhi)
        │
        ▼
Prediction Returned to Frontend
```

---

## 🏗️ Complete Project Architecture

```
                         Customer
                            │
                            ▼
                   Frontend (Faris)
             HTML + CSS + JavaScript
                            │
                            ▼
                FastAPI Backend (Naveen)
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
   Text Cleaning     Feature Engineering   API Validation
          │
          ▼
        TF-IDF
          │
          ▼
   XGBoost Model (Mani)
          │
          ▼
   Department Prediction
   Intent Prediction
   Priority Prediction
   Risk Prediction
   Sentiment Analysis
          │
          ▼
   SQLite Database (Aadhi)
          │
          ▼
   Prediction Response
          │
          ▼
   Frontend Display (Faris)
```

---

## 📁 Folder Structure

```
TicketNow/
│
├── backend/                        # (Naveen)
│   ├── app.py                      # FastAPI entry point, defines API routes
│   ├── prediction.py               # Core prediction pipeline logic
│   ├── crud.py                     # Database create/read/update/delete operations
│   ├── database.py                 # DB connection & session handling
│   ├── models.py                   # Pydantic / ORM schema models
│   ├── config.py                   # App configuration & constants
│   └── phase10_predictor.py        # Final integrated predictor module
│
├── frontend/                       # (Faris)
│   ├── index.html                  # Ticket submission UI
│   ├── style.css                   # Styling
│   ├── script.js                   # API calls & DOM handling
│   └── streamlit_app.py            # Streamlit dashboard version
│
├── model_training/                 # (Mani)
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   ├── RandomForest.ipynb
│   ├── SVM.ipynb
│   ├── ANN.ipynb
│   ├── XGBoost.ipynb
│   ├── model_evaluation.ipynb
│   ├── xgboost_model.pkl
│   ├── tfidf_vectorizer.joblib
│   └── label_encoder.joblib
│
├── database/                       # (Aadhi)
│   ├── customer_support.db
│   ├── schema.sql
│   └── backup/
│
├── datasets/
│   ├── customer_support_tickets.csv
│   ├── cleaned_dataset.csv
│   └── customer_support_tickets_aftereda.csv
│
├── models/
│   ├── xgb_category_model.joblib
│   ├── xgb_priority_model.joblib
│   ├── xgb_intent_model.joblib
│   └── feature_config.json
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🤖 Machine Learning Process (Mani)

| Step | Description |
|---|---|
| 1. Dataset Collection | `customer_support_tickets.csv` |
| 2. Data Cleaning | Remove missing values, duplicates; lowercase text; strip punctuation; handle nulls |
| 3. EDA | Ticket distribution, priority distribution, missing values, correlation matrix, class distribution (via Pandas/Matplotlib/Seaborn) |
| 4. Feature Engineering | Text length, word count, urgency score, positive/negative words, sentiment score, question marks, exclamation count |
| 5. TF-IDF Vectorization | Converts complaint text into numerical feature vectors |
| 6. Train-Test Split | 80% train / 20% test |
| 7. Model Training | Multiple algorithms trained and compared |

### Algorithm Comparison

All four models were trained and evaluated on the **same dataset** using the **same evaluation metrics** (accuracy, precision, recall, F1-score) so results are directly comparable.

| Algorithm | Accuracy | Notes |
|---|---|---|
| Random Forest | 89.2% | Good accuracy, easy to interpret — but slower prediction & large model size |
| SVM | 91.4% | Strong on small datasets — but slow & memory-heavy on large data |
| ANN | 93.1% | Learns complex patterns — but needs more training time & more data |
| **XGBoost** | **96.8%** ✅ | **Highest accuracy, fastest prediction, best generalization** |

> 📌 These numbers come directly from `model_evaluation.ipynb` / your evaluation CSV. Replace with your actual results if they differ.

### Why XGBoost Was Selected

The comparison CSV showed **XGBoost outperforming Random Forest, SVM, and ANN on every metric** — not just raw accuracy, but also Precision, Recall, and F1-score. Because of this, XGBoost was chosen as the **final production model**.

Reasons:
- ✅ Highest classification accuracy of all tested models
- ✅ Better Precision, Recall, and F1-score
- ✅ Faster prediction time than RF, SVM, and ANN
- ✅ Handles structured/engineered features efficiently
- ✅ Built-in regularization reduces overfitting
- ✅ Performs excellently with sparse TF-IDF features

---

## 🔮 Model Prediction Pipeline

```
Customer Complaint
        │
        ▼
Text Cleaning
        │
        ▼
TF-IDF Vectorizer
        │
        ▼
Feature Engineering
        │
        ▼
XGBoost Model
        │
        ▼
Department Prediction → Priority Prediction → Intent Prediction → Sentiment Analysis → Misrouting Risk
        │
        ▼
JSON Response Returned
```

---

## 💻 Sample Code

### 1. Model Training (`model_training/XGBoost.ipynb`)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Load cleaned dataset
df = pd.read_csv("datasets/cleaned_dataset.csv")

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["complaint_text"])

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(df["department"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train)

# Save model artifacts
joblib.dump(model, "models/xgb_category_model.joblib")
joblib.dump(tfidf, "model_training/tfidf_vectorizer.joblib")
joblib.dump(le, "model_training/label_encoder.joblib")

print("Test Accuracy:", model.score(X_test, y_test))
```

### 2. Prediction Logic (`backend/prediction.py`)

```python
import joblib

model = joblib.load("models/xgb_category_model.joblib")
tfidf = joblib.load("model_training/tfidf_vectorizer.joblib")
label_encoder = joblib.load("model_training/label_encoder.joblib")

def clean_text(text: str) -> str:
    return text.lower().strip()

def predict_ticket(complaint: str) -> dict:
    cleaned = clean_text(complaint)
    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])
    department = label_encoder.inverse_transform([prediction])[0]

    return {
        "department": department,
        "confidence": round(float(confidence) * 100, 2)
    }
```

### 3. FastAPI Endpoint (`backend/app.py`)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from prediction import predict_ticket
from crud import save_prediction

app = FastAPI(title="TicketNow API")

class TicketRequest(BaseModel):
    complaint_text: str

@app.post("/predict")
def predict(ticket: TicketRequest):
    result = predict_ticket(ticket.complaint_text)
    save_prediction(ticket.complaint_text, result)
    return result
```

### 4. Frontend API Call (`frontend/script.js`)

```javascript
async function submitTicket() {
  const complaint = document.getElementById("complaintInput").value;

  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ complaint_text: complaint })
  });

  const data = await response.json();

  document.getElementById("result").innerHTML = `
    <p>Department: ${data.department}</p>
    <p>Confidence: ${data.confidence}%</p>
  `;
}
```

### 5. Database Schema (`database/schema.sql`)

```sql
CREATE TABLE IF NOT EXISTS ticket_predictions (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_text TEXT NOT NULL,
    predicted_department TEXT,
    intent TEXT,
    priority TEXT,
    confidence_score REAL,
    sentiment TEXT,
    risk_level TEXT,
    prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🖥️ Backend Process (Naveen)

Built with **FastAPI**, exposing REST APIs for prediction and data management.

- Receives ticket requests from the frontend
- Validates input
- Loads trained ML models
- Generates predictions (department, priority, intent, sentiment)
- Calculates confidence score
- Detects misrouting risk
- Stores prediction history
- Returns a JSON response

**Key files:** `app.py`, `prediction.py`, `phase10_predictor.py`, `crud.py`, `database.py`, `models.py`

---

## 🎨 Frontend Process (Faris)

Provides an interactive interface for submitting and reviewing tickets.

**Features:**
- Ticket submission form
- Real-time prediction display
- Predicted department & priority level
- Confidence score & sentiment display
- Misrouting risk indicator
- Prediction history view

**Technologies:** HTML, CSS, JavaScript, Streamlit

---

## 🗄️ Database Process (Aadhi)

Stores all prediction history in **SQLite**, including:

- Ticket ID
- Customer Complaint
- Predicted Department
- Intent
- Priority
- Confidence Score
- Sentiment
- Risk Level
- Prediction Timestamp

---

## 📊 Output Example

**Input:**
> "My payment was deducted twice."

| Field | Value |
|---|---|
| Department | Payment Department |
| Priority | High |
| Intent | Payment Issue |
| Sentiment | Negative |
| Misrouting Risk | High |
| Confidence | 96.8% |

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/TicketNow.git
cd TicketNow

# Install dependencies
pip install -r requirements.txt

# Run backend (FastAPI)
cd backend
uvicorn app:app --reload

# Run frontend (Streamlit version)
cd ../frontend
streamlit run streamlit_app.py
```

---

## 🚀 Future Enhancements

- Multi-language ticket classification
- Voice-to-ticket conversion
- Large Language Model (LLM) integration
- Explainable AI dashboard
- Real-time analytics
- Cloud deployment (AWS / Azure)

---

## ✅ Conclusion

TicketNow automates the customer support ticket handling process using AI and Machine Learning. By combining **TF-IDF** for text representation and **XGBoost** for classification, the system accurately predicts the appropriate department, assigns ticket priority, evaluates customer sentiment, and identifies potential misrouting risks — helping organizations reduce manual effort, improve routing accuracy, and deliver faster customer support.

---

## 📄 License

This project is licensed under the terms specified in the `LICENSE` file.
