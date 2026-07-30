import sys, os
sys.path.insert(0, r'C:\Users\acer\Downloads\TicketNow-20260723T135840Z-1-001\TicketNow\backend')
from ml_engine import MLEngine
from scipy.sparse import hstack, csr_matrix

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

print("=== STANDARD MODEL PREDICTIONS (without pkl override) ===")
for i, text in enumerate(complaints, 1):
    result = engine.predict(text)
    dept = result['category']
    conf = result.get('confidence', 0)
    clean = engine._clean_text(text)
    X_tfidf = engine.tfidf.transform([clean])
    eng_features = engine._engineer_features(text)
    feature_order = ['text_length', 'word_count', 'avg_word_length',
                    'urgency_score', 'negative_word_count', 'positive_word_count',
                    'sentiment_score', 'has_question', 'exclamation_count']
    X_eng = csr_matrix([[eng_features.get(f, 0) for f in feature_order]])
    X_combined = hstack([X_tfidf, X_eng])
    y_pred_cat = engine.xgb_category.predict(X_combined)
    cat_proba = engine.xgb_category.predict_proba(X_combined)[0]
    cat_idx = int(y_pred_cat[0])
    std_dept = engine.le_category.inverse_transform([cat_idx])[0]
    std_confidence = float(cat_proba[cat_idx])
    print(f'Test {i}: {text}')
    print(f'  PKL Model: {dept} ({conf:.4f})')
    print(f'  Standard Model: {std_dept} ({std_confidence:.4f})')
    print()
