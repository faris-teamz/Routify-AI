import sys, os
sys.path.insert(0, r'C:\Users\acer\Downloads\TicketNow-20260723T135840Z-1-001\TicketNow\backend')
from ml_engine import MLEngine

model_dir = r'C:\Users\acer\Downloads\TicketNow-20260723T135840Z-1-001\TicketNow\EDA AND MODEL\trained_models'
engine = MLEngine(model_dir)

complaints = [
    'I was charged twice for the same order.',
    'Refund was requested but I have not received the amount yet.',
    'My account repeatedly logs me out whenever I try to check my transaction history.',
    'An error appears whenever I try to access my payment history.',
    'I already contacted support about my issue, but I have not received a proper response.',
    'My complaint has remained unresolved for several days and I still have not received a solution.',
    'I need this payment issue investigated and resolved immediately.',
    'I was charged twice for one order, the refund is still pending, and now I am unable to access my account because of repeated login errors.',
]

for i, text in enumerate(complaints, 1):
    result = engine.predict(text)
    dept = result['category']
    conf = result['confidence']
    pri = result['priority']
    risk = result['risk_level']
    intent = result['intent']
    human = 1 if risk == 'High' else 0
    print(f'Test {i}: {text}')
    print(f'  Department: {dept}')
    print(f'  Confidence: {conf:.4f}')
    print(f'  Priority: {pri}')
    print(f'  Risk Level: {risk}')
    print(f'  Intent: {intent}')
    print(f'  Requires Human: {human}')
    print()
