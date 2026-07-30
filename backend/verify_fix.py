import sys, os
sys.path.insert(0, r'C:\Users\acer\Downloads\TicketNow-20260723T135840Z-1-001\TicketNow\backend')
from ml_engine import MLEngine

model_dir = r'C:\Users\acer\Downloads\TicketNow-20260723T135840Z-1-001\TicketNow\EDA AND MODEL\trained_models'
engine = MLEngine(model_dir)

print("=== Verification ===")
print(f"ML Engine loaded: {engine.is_loaded()}")
print(f"PKL model loaded: {engine.is_pkl_loaded()}")
print(f"Model dir: {engine.model_dir}")
print(f"PKL path: {engine.pkl_model_path}")

# Quick prediction test
result = engine.predict("I was charged twice for the same order.")
print(f"\nTest prediction:")
print(f"  Category: {result['category']}")
print(f"  Confidence: {result['confidence']:.4f}")
print(f"  Priority: {result['priority']}")
print(f"  Risk Level: {result['risk_level']}")
