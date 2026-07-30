"""
TicketNow - ML Model Training
===============================
Trains XGBoost classifiers for:
1. Category/Department prediction (27 intents → 10 categories)
2. Priority prediction
3. Misrouting Risk Engine

Uses TF-IDF features + engineered features for high accuracy.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'trained_models')
PLOTS_DIR = os.path.join(BASE_DIR, 'eda_plots')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

print("=" * 60)
print("TICKETNOW - MODEL TRAINING PIPELINE")
print("=" * 60)

# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================
print("\n[1/7] Loading processed dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, 'processed_dataset.csv'))

# Load config
with open(os.path.join(BASE_DIR, 'eda_config.json'), 'r') as f:
    config = json.load(f)

text_col = config['text_column']
intent_col = config['intent_column']
category_col = config['category_column']

print(f"   Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Text: '{text_col}', Intent: '{intent_col}', Category: '{category_col}'")

# ============================================================
# 2. TF-IDF VECTORIZATION
# ============================================================
print("\n[2/7] TF-IDF Vectorization...")

# Clean text
df['clean_text'] = df[text_col].astype(str).str.lower()
df['clean_text'] = df['clean_text'].str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
df['clean_text'] = df['clean_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

# TF-IDF with optimized parameters
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams + Bigrams
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents='unicode'
)

X_tfidf = tfidf.fit_transform(df['clean_text'])
print(f"   TF-IDF shape: {X_tfidf.shape}")
print(f"   Vocabulary size: {len(tfidf.vocabulary_)}")

# Show top features
feature_names = tfidf.get_feature_names_out()
tfidf_scores = X_tfidf.sum(axis=0).A1
top_indices = tfidf_scores.argsort()[-20:][::-1]
print(f"\n   Top 20 TF-IDF Features:")
for i, idx in enumerate(top_indices):
    print(f"   {i+1:2d}. {feature_names[idx]:25s} (score: {tfidf_scores[idx]:.2f})")

# Combine TF-IDF with engineered features
engineered_features = ['text_length', 'word_count', 'avg_word_length', 
                       'urgency_score', 'negative_word_count', 'positive_word_count',
                       'sentiment_score', 'has_question', 'exclamation_count']
available_features = [f for f in engineered_features if f in df.columns]

X_eng = csr_matrix(df[available_features].values.astype(float))
X_combined = hstack([X_tfidf, X_eng])
print(f"\n   Combined features shape: {X_combined.shape}")
print(f"   (TF-IDF: {X_tfidf.shape[1]} + Engineered: {X_eng.shape[1]})")

# ============================================================
# 3. ENCODE LABELS
# ============================================================
print("\n[3/7] Encoding Labels...")

# Category encoder
le_category = LabelEncoder()
y_category = le_category.fit_transform(df[category_col])
print(f"   Categories ({len(le_category.classes_)}): {list(le_category.classes_)}")

# Intent encoder (finer-grained)
le_intent = LabelEncoder()
y_intent = le_intent.fit_transform(df[intent_col])
print(f"   Intents ({len(le_intent.classes_)}): {list(le_intent.classes_)}")

# Priority encoder
le_priority = LabelEncoder()
y_priority = le_priority.fit_transform(df['predicted_priority'])
print(f"   Priorities ({len(le_priority.classes_)}): {list(le_priority.classes_)}")

# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================
print("\n[4/7] Train/Test Split...")

# Split for category prediction
X_train, X_test, y_train_cat, y_test_cat = train_test_split(
    X_combined, y_category, test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, stratify=y_category
)

# Corresponding intent and priority labels
_, _, y_train_intent, y_test_intent = train_test_split(
    X_combined, y_intent, test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, stratify=y_category
)

_, _, y_train_priority, y_test_priority = train_test_split(
    X_combined, y_priority, test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, stratify=y_category
)

print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test:  {X_test.shape[0]} samples")

# ============================================================
# 5. MODEL TRAINING
# ============================================================
print("\n[5/7] Training Models...")
print("-" * 40)

results = {}

# ----- Model 1: XGBoost Category Classifier -----
print("\n   [TOOL] Training XGBoost Category Classifier...")
xgb_cat = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='multi:softprob',
    num_class=len(le_category.classes_),
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='mlogloss',
    use_label_encoder=False
)
xgb_cat.fit(X_train, y_train_cat)
y_pred_cat = xgb_cat.predict(X_test)
y_prob_cat = xgb_cat.predict_proba(X_test)

cat_accuracy = accuracy_score(y_test_cat, y_pred_cat)
cat_f1 = f1_score(y_test_cat, y_pred_cat, average='weighted')
print(f"      Accuracy: {cat_accuracy:.4f}")
print(f"      F1 Score: {cat_f1:.4f}")
results['xgb_category'] = {'accuracy': cat_accuracy, 'f1': cat_f1}

# ----- Model 2: XGBoost Intent Classifier (Fine-grained) -----
print("\n   [TOOL] Training XGBoost Intent Classifier...")
xgb_intent = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.05,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='multi:softprob',
    num_class=len(le_intent.classes_),
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='mlogloss',
    use_label_encoder=False
)
xgb_intent.fit(X_train, y_train_intent)
y_pred_intent = xgb_intent.predict(X_test)
y_prob_intent = xgb_intent.predict_proba(X_test)

intent_accuracy = accuracy_score(y_test_intent, y_pred_intent)
intent_f1 = f1_score(y_test_intent, y_pred_intent, average='weighted')
print(f"      Accuracy: {intent_accuracy:.4f}")
print(f"      F1 Score: {intent_f1:.4f}")
results['xgb_intent'] = {'accuracy': intent_accuracy, 'f1': intent_f1}

# ----- Model 3: Logistic Regression Baseline (Category) -----
print("\n   [TOOL] Training Logistic Regression Baseline...")
lr_cat = LogisticRegression(
    max_iter=1000, 
    C=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    multi_class='multinomial',
    solver='lbfgs'
)
lr_cat.fit(X_train, y_train_cat)
y_pred_lr = lr_cat.predict(X_test)
y_prob_lr = lr_cat.predict_proba(X_test)

lr_accuracy = accuracy_score(y_test_cat, y_pred_lr)
lr_f1 = f1_score(y_test_cat, y_pred_lr, average='weighted')
print(f"      Accuracy: {lr_accuracy:.4f}")
print(f"      F1 Score: {lr_f1:.4f}")
results['lr_baseline'] = {'accuracy': lr_accuracy, 'f1': lr_f1}

# ----- Model 4: XGBoost Priority Classifier -----
print("\n   [TOOL] Training XGBoost Priority Classifier...")
xgb_priority = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=len(le_priority.classes_),
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='mlogloss',
    use_label_encoder=False
)
xgb_priority.fit(X_train, y_train_priority)
y_pred_priority = xgb_priority.predict(X_test)
y_prob_priority = xgb_priority.predict_proba(X_test)

priority_accuracy = accuracy_score(y_test_priority, y_pred_priority)
priority_f1 = f1_score(y_test_priority, y_pred_priority, average='weighted')
print(f"      Accuracy: {priority_accuracy:.4f}")
print(f"      F1 Score: {priority_f1:.4f}")
results['xgb_priority'] = {'accuracy': priority_accuracy, 'f1': priority_f1}

# ============================================================
# 6. EVALUATION & VISUALIZATIONS
# ============================================================
print("\n[6/7] Evaluation & Visualizations...")

# ----- Classification Reports -----
print("\n   [REPORT] XGBoost Category Classification Report:")
print(classification_report(y_test_cat, y_pred_cat, 
                           target_names=le_category.classes_))

print("\n   [REPORT] XGBoost Intent Classification Report:")
print(classification_report(y_test_intent, y_pred_intent, 
                           target_names=le_intent.classes_))

# ----- Plot: Model Comparison -----
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
f1_scores_list = [results[m]['f1'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35
bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#00D4FF', alpha=0.8)
bars2 = ax.bar(x + width/2, f1_scores_list, width, label='F1 Score', color='#7B68EE', alpha=0.8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['XGB\nCategory', 'XGB\nIntent', 'LR\nBaseline', 'XGB\nPriority'],
                    fontsize=11)
ax.legend(fontsize=12)
ax.set_ylim(0, 1.1)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{bar.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{bar.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '11_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] Saved: 11_model_comparison.png")

# ----- Plot: Category Confusion Matrix -----
fig, ax = plt.subplots(figsize=(12, 10))
cm = confusion_matrix(y_test_cat, y_pred_cat)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_category.classes_, 
            yticklabels=le_category.classes_, ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Category Confusion Matrix (XGBoost)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '12_confusion_matrix_category.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] Saved: 12_confusion_matrix_category.png")

# ----- Plot: Intent Confusion Matrix -----
fig, ax = plt.subplots(figsize=(16, 14))
cm_intent = confusion_matrix(y_test_intent, y_pred_intent)
sns.heatmap(cm_intent, annot=True, fmt='d', cmap='Purples',
            xticklabels=le_intent.classes_, 
            yticklabels=le_intent.classes_, ax=ax,
            annot_kws={"size": 7})
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Intent Confusion Matrix (XGBoost - 27 Classes)', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '13_confusion_matrix_intent.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] Saved: 13_confusion_matrix_intent.png")

# ----- Plot: Confidence Distribution -----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Category confidence
max_probs_cat = y_prob_cat.max(axis=1)
axes[0].hist(max_probs_cat, bins=50, color='#00D4FF', edgecolor='white', alpha=0.8)
axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Triage Threshold (0.5)')
axes[0].set_xlabel('Max Prediction Probability')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Category Prediction Confidence', fontsize=13, fontweight='bold')
axes[0].legend()

# Intent confidence
max_probs_intent = y_prob_intent.max(axis=1)
axes[1].hist(max_probs_intent, bins=50, color='#7B68EE', edgecolor='white', alpha=0.8)
axes[1].axvline(0.4, color='red', linestyle='--', linewidth=2, label='Triage Threshold (0.4)')
axes[1].set_xlabel('Max Prediction Probability')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Intent Prediction Confidence', fontsize=13, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '14_confidence_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] Saved: 14_confidence_distribution.png")

# ----- Misrouting Risk Analysis -----
print("\n   [SEARCH] Misrouting Risk Analysis:")
# Calculate risk metrics
confidence_threshold_cat = 0.5
confidence_threshold_intent = 0.4

# Probability gap between top-2 predictions (ambiguity measure)
top2_gaps_cat = np.sort(y_prob_cat, axis=1)[:, -1] - np.sort(y_prob_cat, axis=1)[:, -2]
top2_gaps_intent = np.sort(y_prob_intent, axis=1)[:, -1] - np.sort(y_prob_intent, axis=1)[:, -2]

low_conf_cat = (max_probs_cat < confidence_threshold_cat).sum()
low_conf_intent = (max_probs_intent < confidence_threshold_intent).sum()
ambiguous = (top2_gaps_cat < 0.15).sum()

print(f"   Category low confidence (<{confidence_threshold_cat}): {low_conf_cat} ({low_conf_cat/len(max_probs_cat)*100:.1f}%)")
print(f"   Intent low confidence (<{confidence_threshold_intent}): {low_conf_intent} ({low_conf_intent/len(max_probs_intent)*100:.1f}%)")
print(f"   Ambiguous (top-2 gap < 0.15): {ambiguous} ({ambiguous/len(top2_gaps_cat)*100:.1f}%)")

# ----- Plot: Risk Level Distribution -----
fig, ax = plt.subplots(figsize=(8, 6))

# Compute risk scores
risk_scores = []
for i in range(len(max_probs_cat)):
    conf = max_probs_cat[i]
    gap = top2_gaps_cat[i]
    
    # Risk formula: lower confidence + smaller gap = higher risk
    risk = (1 - conf) * 0.6 + (1 - min(gap * 2, 1)) * 0.4
    
    if risk > 0.6:
        risk_scores.append('High')
    elif risk > 0.35:
        risk_scores.append('Medium')
    else:
        risk_scores.append('Low')

risk_df = pd.Series(risk_scores)
risk_counts = risk_df.value_counts()
colors = {'Low': '#4CAF50', 'Medium': '#FFD700', 'High': '#FF4444'}
bars = ax.bar(risk_counts.index, risk_counts.values, 
              color=[colors[r] for r in risk_counts.index])
ax.set_ylabel('Number of Tickets')
ax.set_title('Misrouting Risk Distribution', fontsize=14, fontweight='bold')
for bar, val in zip(bars, risk_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            f'{val}\n({val/len(risk_scores)*100:.1f}%)', 
            ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '15_risk_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] Saved: 15_risk_distribution.png")

# ----- Cross Validation -----
print("\n   [LOOP] Cross-Validation (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(xgb_cat, X_combined, y_category, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"   CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"   Fold scores: {[f'{s:.4f}' for s in cv_scores]}")

# ============================================================
# 7. SAVE MODELS & ARTIFACTS
# ============================================================
print("\n[7/7] Saving Models & Artifacts...")

# Save models
joblib.dump(xgb_cat, os.path.join(MODEL_DIR, 'xgb_category_model.joblib'))
joblib.dump(xgb_intent, os.path.join(MODEL_DIR, 'xgb_intent_model.joblib'))
joblib.dump(xgb_priority, os.path.join(MODEL_DIR, 'xgb_priority_model.joblib'))
joblib.dump(lr_cat, os.path.join(MODEL_DIR, 'lr_baseline_model.joblib'))
print("   [SAVE] Saved: xgb_category_model.joblib")
print("   [SAVE] Saved: xgb_intent_model.joblib")
print("   [SAVE] Saved: xgb_priority_model.joblib")
print("   [SAVE] Saved: lr_baseline_model.joblib")

# Save TF-IDF vectorizer
joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
print("   [SAVE] Saved: tfidf_vectorizer.joblib")

# Save label encoders
joblib.dump(le_category, os.path.join(MODEL_DIR, 'le_category.joblib'))
joblib.dump(le_intent, os.path.join(MODEL_DIR, 'le_intent.joblib'))
joblib.dump(le_priority, os.path.join(MODEL_DIR, 'le_priority.joblib'))
print("   [SAVE] Saved: le_category.joblib, le_intent.joblib, le_priority.joblib")

# Save model performance report
model_report = {
    'models': results,
    'cross_validation': {
        'mean_accuracy': float(cv_scores.mean()),
        'std': float(cv_scores.std()),
        'fold_scores': [float(s) for s in cv_scores]
    },
    'misrouting_risk': {
        'low_confidence_category_pct': float(low_conf_cat / len(max_probs_cat) * 100),
        'low_confidence_intent_pct': float(low_conf_intent / len(max_probs_intent) * 100),
        'ambiguous_tickets_pct': float(ambiguous / len(top2_gaps_cat) * 100),
        'risk_distribution': risk_counts.to_dict()
    },
    'feature_info': {
        'tfidf_features': int(X_tfidf.shape[1]),
        'engineered_features': available_features,
        'total_features': int(X_combined.shape[1])
    },
    'categories': list(le_category.classes_),
    'intents': list(le_intent.classes_),
    'priorities': list(le_priority.classes_),
    'thresholds': {
        'category_confidence': confidence_threshold_cat,
        'intent_confidence': confidence_threshold_intent,
        'ambiguity_gap': 0.15,
        'risk_high': 0.6,
        'risk_medium': 0.35
    }
}

with open(os.path.join(MODEL_DIR, 'model_report.json'), 'w') as f:
    json.dump(model_report, f, indent=2)
print("   [SAVE] Saved: model_report.json")

# Save engineered features list for the backend
with open(os.path.join(MODEL_DIR, 'feature_config.json'), 'w') as f:
    json.dump({
        'text_column': text_col,
        'engineered_features': available_features,
        'urgency_keywords': ['urgent', 'immediately', 'asap', 'emergency', 'critical',
                             'right now', 'help me', 'desperate', 'not working', 'broken',
                             'cannot', "can't", 'unable', 'failed', 'error', 'crash'],
        'negative_words': ['angry', 'frustrated', 'terrible', 'worst', 'horrible', 'hate',
                           'disappointed', 'unacceptable', 'ridiculous', 'waste', 'scam',
                           'refuse', 'complaint', 'problem', 'issue', 'wrong', 'bad'],
        'positive_words': ['thank', 'please', 'appreciate', 'great', 'good', 'help',
                           'kind', 'wonderful', 'excellent', 'love', 'happy', 'satisfied']
    }, f, indent=2)
print("   [SAVE] Saved: feature_config.json")

print("\n" + "=" * 60)
print("[OK] MODEL TRAINING COMPLETE!")
print(f"   [DIR] Models saved to: {MODEL_DIR}")
print(f"\n   [CHART] RESULTS SUMMARY:")
print(f"   {'Model':<25s} {'Accuracy':>10s} {'F1 Score':>10s}")
print(f"   {'-'*45}")
for model_name, metrics in results.items():
    print(f"   {model_name:<25s} {metrics['accuracy']:>10.4f} {metrics['f1']:>10.4f}")
print(f"\n   [CHART] Cross-Validation: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print("=" * 60)
