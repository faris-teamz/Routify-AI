import sys, os
sys.path.insert(0, r'C:\Users\acer\Downloads\TicketNow-20260723T135840Z-1-001\TicketNow\backend')
from ml_engine import MLEngine

model_dir = r'C:\Users\acer\Downloads\TicketNow-20260723T135840Z-1-001\TicketNow\EDA AND MODEL\trained_models'
engine = MLEngine(model_dir)

tests = [
    "I was charged twice for the same order.",
    "An error appears whenever I try to access my payment history.",
    "I need this payment issue investigated and resolved immediately.",
    "My payment was deducted but the order shows as failed. I need an immediate refund.",
    "The app is extremely slow and keeps crashing when I try to open the dashboard.",
    "I can't log into my account. It keeps showing 'invalid credentials' even though I'm using the right password.",
    "There's a bug in the report generation feature. Numbers don't add up correctly.",
    "I received an unauthorized login alert. Someone might have access to my account.",
]

for text in tests:
    result = engine.predict(text)
    print(f"Complaint: {text}")
    print(f"  Category: {result['category']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Ambiguity Gap: {result['ambiguity_gap']:.4f}")
    print(f"  Priority: {result['priority']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Risk Score: {result['risk_score']:.4f}")
    print(f"  Probs: {result['category_probabilities']}")
    print()
