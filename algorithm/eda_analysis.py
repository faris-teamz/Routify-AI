"""
TicketNow - EDA & Data Preprocessing
=====================================
Downloads the Bitext Customer Support dataset and performs comprehensive EDA.
Generates visualizations and preprocessed data for model training.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DOWNLOAD & LOAD DATASET
# ============================================================
print("=" * 60)
print("TICKETNOW - EDA & DATA PREPROCESSING")
print("=" * 60)

# Try loading from HuggingFace datasets library
print("\n[1/8] Loading Bitext Customer Support Dataset...")
try:
    from datasets import load_dataset
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = ds['train'].to_pandas()
    print(f"   [OK] Loaded from HuggingFace: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"   [WARN] HuggingFace download failed: {e}")
    print("   Trying alternative method...")
    try:
        # Fallback: try reading from local if already downloaded
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'bitext_customer_support.csv'))
        print(f"   [OK] Loaded from local CSV: {df.shape[0]} rows, {df.shape[1]} columns")
    except:
        print("   [ERR] Could not load dataset. Please download manually.")
        exit(1)

# Save raw dataset locally for future use
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'eda_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

df.to_csv(os.path.join(OUTPUT_DIR, 'bitext_customer_support.csv'), index=False)
print(f"   [SAVE] Saved raw dataset to bitext_customer_support.csv")

# ============================================================
# 2. BASIC DATA OVERVIEW
# ============================================================
print("\n[2/8] Basic Data Overview")
print("-" * 40)
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"\n   Data Types:")
print(df.dtypes.to_string())
print(f"\n   Missing Values:")
print(df.isnull().sum().to_string())
print(f"\n   First 3 Rows:")
print(df.head(3).to_string())

# ============================================================
# 3. TEXT COLUMN IDENTIFICATION
# ============================================================
print("\n[3/8] Identifying Key Columns...")

# Identify the main text column and label columns
text_col = None
for col in ['instruction', 'text', 'query', 'question', 'complaint', 'description']:
    if col in df.columns:
        text_col = col
        break

if text_col is None:
    # Use the first string column with longest avg text
    str_cols = df.select_dtypes(include='object').columns
    avg_lens = {col: df[col].astype(str).str.len().mean() for col in str_cols}
    text_col = max(avg_lens, key=avg_lens.get)

intent_col = None
for col in ['intent', 'label', 'class', 'target']:
    if col in df.columns:
        intent_col = col
        break

category_col = None
for col in ['category', 'department', 'group', 'topic']:
    if col in df.columns:
        category_col = col
        break

print(f"   Text column: '{text_col}'")
print(f"   Intent column: '{intent_col}'")
print(f"   Category column: '{category_col}'")

# ============================================================
# 4. UNIQUENESS ANALYSIS (Critical check!)
# ============================================================
print("\n[4/8] Uniqueness Analysis (Template Check)")
print("-" * 40)
total_texts = len(df[text_col])
unique_texts = df[text_col].nunique()
uniqueness_ratio = unique_texts / total_texts * 100
print(f"   Total texts: {total_texts}")
print(f"   Unique texts: {unique_texts}")
print(f"   Uniqueness ratio: {uniqueness_ratio:.1f}%")

if uniqueness_ratio > 80:
    print("   [OK] EXCELLENT: High text diversity — suitable for ML!")
elif uniqueness_ratio > 50:
    print("   [WARN] MODERATE: Some repetition but usable")
else:
    print("   [ERR] WARNING: High repetition — may be template-based")

# Show sample texts
print(f"\n   Sample Texts (first 5 unique):")
for i, txt in enumerate(df[text_col].unique()[:5]):
    print(f"   {i+1}. {txt[:100]}...")

# ============================================================
# 5. CATEGORY & INTENT DISTRIBUTION
# ============================================================
print("\n[5/8] Category & Intent Distribution")
print("-" * 40)

if category_col:
    print(f"\n   Categories ({df[category_col].nunique()} unique):")
    print(df[category_col].value_counts().to_string())

if intent_col:
    print(f"\n   Intents ({df[intent_col].nunique()} unique):")
    print(df[intent_col].value_counts().to_string())

# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================
print("\n[6/8] Feature Engineering...")

# Text length features
df['text_length'] = df[text_col].astype(str).str.len()
df['word_count'] = df[text_col].astype(str).str.split().str.len()
df['avg_word_length'] = df[text_col].astype(str).apply(
    lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
)

# Urgency keywords detection
urgency_keywords = ['urgent', 'immediately', 'asap', 'emergency', 'critical', 
                    'right now', 'help me', 'desperate', 'not working', 'broken',
                    'cannot', "can't", 'unable', 'failed', 'error', 'crash']
df['urgency_score'] = df[text_col].astype(str).str.lower().apply(
    lambda x: sum(1 for kw in urgency_keywords if kw in x)
)

# Sentiment keywords (simple rule-based for now)
negative_words = ['angry', 'frustrated', 'terrible', 'worst', 'horrible', 'hate',
                  'disappointed', 'unacceptable', 'ridiculous', 'waste', 'scam',
                  'refuse', 'complaint', 'problem', 'issue', 'wrong', 'bad']
positive_words = ['thank', 'please', 'appreciate', 'great', 'good', 'help',
                  'kind', 'wonderful', 'excellent', 'love', 'happy', 'satisfied']

df['negative_word_count'] = df[text_col].astype(str).str.lower().apply(
    lambda x: sum(1 for w in negative_words if w in x)
)
df['positive_word_count'] = df[text_col].astype(str).str.lower().apply(
    lambda x: sum(1 for w in positive_words if w in x)
)
df['sentiment_score'] = df['positive_word_count'] - df['negative_word_count']

# Priority assignment based on urgency
def assign_priority(row):
    if row['urgency_score'] >= 3:
        return 'Critical'
    elif row['urgency_score'] >= 2:
        return 'High'
    elif row['urgency_score'] >= 1:
        return 'Medium'
    else:
        return 'Low'

df['predicted_priority'] = df.apply(assign_priority, axis=1)

# Question detection
df['has_question'] = df[text_col].astype(str).str.contains(r'\?', regex=True).astype(int)

# Exclamation detection (possible frustration)
df['exclamation_count'] = df[text_col].astype(str).str.count('!')

print("   [OK] Engineered features: text_length, word_count, avg_word_length,")
print("      urgency_score, sentiment_score, predicted_priority, has_question, exclamation_count")

print(f"\n   Priority Distribution (engineered):")
print(df['predicted_priority'].value_counts().to_string())

print(f"\n   Feature Statistics:")
print(df[['text_length', 'word_count', 'urgency_score', 'sentiment_score']].describe().to_string())

# ============================================================
# 7. GENERATE EDA VISUALIZATIONS
# ============================================================
print("\n[7/8] Generating EDA Visualizations...")

# Set global plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = ['#00D4FF', '#7B68EE', '#FF6B6B', '#4ECDC4', '#45B7D1', 
          '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
          '#BB8FCE', '#85C1E9', '#F1948A', '#82E0AA', '#F8C471',
          '#AED6F1', '#D7BDE2', '#A3E4D7', '#F9E79F', '#FADBD8',
          '#D5F5E3', '#E8DAEF', '#D6EAF8', '#FCF3CF', '#FDEDEC',
          '#E8F8F5', '#FDF2E9']

# ----- Plot 1: Category Distribution -----
if category_col:
    fig, ax = plt.subplots(figsize=(12, 6))
    cat_counts = df[category_col].value_counts()
    bars = ax.barh(cat_counts.index, cat_counts.values, color=COLORS[:len(cat_counts)])
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title('Category Distribution', fontsize=16, fontweight='bold')
    for bar, val in zip(bars, cat_counts.values):
        ax.text(val + 10, bar.get_y() + bar.get_height()/2, str(val), 
                va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_category_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   [CHART] Saved: 01_category_distribution.png")

# ----- Plot 2: Intent Distribution -----
if intent_col:
    fig, ax = plt.subplots(figsize=(14, 8))
    intent_counts = df[intent_col].value_counts()
    bars = ax.barh(intent_counts.index, intent_counts.values, 
                   color=COLORS[:len(intent_counts)])
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title('Intent Distribution (27 Classes)', fontsize=16, fontweight='bold')
    for bar, val in zip(bars, intent_counts.values):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2, str(val), 
                va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_intent_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   [CHART] Saved: 02_intent_distribution.png")

# ----- Plot 3: Text Length Distribution -----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['text_length'], bins=50, color='#00D4FF', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Character Count')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Text Length Distribution', fontsize=14, fontweight='bold')
axes[0].axvline(df['text_length'].mean(), color='red', linestyle='--', label=f"Mean: {df['text_length'].mean():.0f}")
axes[0].legend()

axes[1].hist(df['word_count'], bins=30, color='#7B68EE', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
axes[1].axvline(df['word_count'].mean(), color='red', linestyle='--', label=f"Mean: {df['word_count'].mean():.0f}")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_text_length_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] Saved: 03_text_length_distribution.png")

# ----- Plot 4: Priority Distribution (Engineered) -----
fig, ax = plt.subplots(figsize=(8, 6))
priority_counts = df['predicted_priority'].value_counts()
priority_colors = {'Critical': '#FF4444', 'High': '#FF8C00', 'Medium': '#FFD700', 'Low': '#4CAF50'}
colors = [priority_colors.get(p, '#888') for p in priority_counts.index]
wedges, texts, autotexts = ax.pie(priority_counts.values, labels=priority_counts.index, 
                                   autopct='%1.1f%%', colors=colors, startangle=90,
                                   textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_fontweight('bold')
ax.set_title('Priority Distribution (Engineered from Text)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_priority_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] Saved: 04_priority_distribution.png")

# ----- Plot 5: Sentiment Score Distribution -----
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['sentiment_score'], bins=range(df['sentiment_score'].min(), df['sentiment_score'].max()+2),
        color='#4ECDC4', edgecolor='white', alpha=0.8)
ax.set_xlabel('Sentiment Score (positive - negative words)')
ax.set_ylabel('Frequency')
ax.set_title('Sentiment Score Distribution', fontsize=14, fontweight='bold')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Neutral')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_sentiment_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] Saved: 05_sentiment_distribution.png")

# ----- Plot 6: Text Length by Category (Box Plot) -----
if category_col:
    fig, ax = plt.subplots(figsize=(14, 6))
    category_order = df.groupby(category_col)['text_length'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, y=category_col, x='text_length', order=category_order,
                palette=COLORS[:len(category_order)], ax=ax)
    ax.set_xlabel('Text Length (characters)')
    ax.set_title('Text Length by Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_text_length_by_category.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   [CHART] Saved: 06_text_length_by_category.png")

# ----- Plot 7: Urgency Score by Category -----
if category_col:
    fig, ax = plt.subplots(figsize=(12, 6))
    urgency_by_cat = df.groupby(category_col)['urgency_score'].mean().sort_values(ascending=False)
    bars = ax.bar(range(len(urgency_by_cat)), urgency_by_cat.values, 
                  color=COLORS[:len(urgency_by_cat)])
    ax.set_xticks(range(len(urgency_by_cat)))
    ax.set_xticklabels(urgency_by_cat.index, rotation=45, ha='right')
    ax.set_ylabel('Average Urgency Score')
    ax.set_title('Average Urgency Score by Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '07_urgency_by_category.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   [CHART] Saved: 07_urgency_by_category.png")

# ----- Plot 8: Feature Correlation Heatmap -----
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = ['text_length', 'word_count', 'avg_word_length', 'urgency_score', 
                'negative_word_count', 'positive_word_count', 'sentiment_score',
                'has_question', 'exclamation_count']
available_cols = [c for c in numeric_cols if c in df.columns]
corr_matrix = df[available_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '08_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   [CHART] Saved: 08_correlation_heatmap.png")

# ----- Plot 9: Word Cloud (Overall) -----
try:
    from wordcloud import WordCloud
    all_text = ' '.join(df[text_col].astype(str).values)
    wc = WordCloud(width=1200, height=600, background_color='#0A0E27',
                   colormap='cool', max_words=150, max_font_size=80,
                   random_state=42)
    wc.generate(all_text)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud — All Tickets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '09_wordcloud_overall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   [CHART] Saved: 09_wordcloud_overall.png")
except ImportError:
    print("   [WARN] wordcloud not installed, skipping word cloud")

# ----- Plot 10: Category vs Priority Heatmap -----
if category_col:
    fig, ax = plt.subplots(figsize=(10, 8))
    ct = pd.crosstab(df[category_col], df['predicted_priority'])
    # Reorder priority columns
    priority_order = ['Critical', 'High', 'Medium', 'Low']
    available_priorities = [p for p in priority_order if p in ct.columns]
    ct = ct[available_priorities]
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax, linewidths=0.5)
    ax.set_title('Category vs Priority (Engineered)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Priority')
    ax.set_ylabel('Category')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '10_category_priority_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   [CHART] Saved: 10_category_priority_heatmap.png")

# ============================================================
# 8. SAVE PROCESSED DATA
# ============================================================
print("\n[8/8] Saving Processed Data...")

# Save the enriched dataframe
df.to_csv(os.path.join(OUTPUT_DIR, 'processed_dataset.csv'), index=False)
print(f"   [SAVE] Saved: processed_dataset.csv ({df.shape[0]} rows, {df.shape[1]} columns)")

# Save column info
col_info = {
    'text_column': text_col,
    'intent_column': intent_col,
    'category_column': category_col,
    'total_rows': len(df),
    'unique_texts': unique_texts,
    'num_categories': df[category_col].nunique() if category_col else 0,
    'num_intents': df[intent_col].nunique() if intent_col else 0,
    'engineered_features': ['text_length', 'word_count', 'avg_word_length', 
                            'urgency_score', 'negative_word_count', 'positive_word_count',
                            'sentiment_score', 'predicted_priority', 'has_question', 'exclamation_count']
}

import json
with open(os.path.join(OUTPUT_DIR, 'eda_config.json'), 'w') as f:
    json.dump(col_info, f, indent=2)
print(f"   [SAVE] Saved: eda_config.json")

print("\n" + "=" * 60)
print("[OK] EDA COMPLETE!")
print(f"   [DIR] Plots saved to: {PLOTS_DIR}")
print(f"   [DIR] Processed data: {os.path.join(OUTPUT_DIR, 'processed_dataset.csv')}")
print("=" * 60)
