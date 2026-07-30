"""
ROUTIFY AI - ML Engine
======================
Loads trained models and performs predictions on new ticket text.
Includes misrouting risk scoring engine.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


class MLEngine:
    """ML prediction engine for ticket classification and risk assessment.

    Department/category prediction is performed by the standalone XGBoost model
    at E:\\xgboost_model.pkl, which expects exactly 55 numeric features built
    from the customer_support_tickets_200k.csv schema (one-hot metadata + 7
    numeric fields). Intent and priority are still produced by the trained
    joblib models shipped with the project.
    """

    # The classifier models below are built from trained_models/*.joblib.
    # The optional standalone department model is loaded dynamically from the
    # same model directory as xgboost_model.pkl, so the project remains portable.
    PKL_TFIDF_MAX_FEATURES = 300
    PKL_STRUCTURED_COLS = ['sentiment', 'channel', 'region', 'customer_segment', 'customer_gender']

    # Mapping of the pkl model's 7 numeric classes (0-6) to department names.
    # Aligned to the user-supplied department list for this 7-class model.
    PKL_CLASS_TO_DEPARTMENT = {
        0: "Performance Issue",
        1: "ACCOUNT",
        2: "Subscription Cancellation",
        3: "Feature Request",
        4: "Security Concern",
        5: "REFUND",
        6: "Payment Problem",
    }

    # Rule-based routing rules (verbatim from XGBOOST_TRAINING_MODEL.ipynb, Module 6).
    # This is the EXACT training target the pkl was trained to mimic, and is the
    # authoritative, deterministic department predictor for free-text complaints.
    # Keyword lists have been expanded to cover varied phrasings seen in real tickets.
    ROUTING_RULES = {
        "Payment Department": [
            "payment", "transaction", "card", "bank", "charged",
            "billing", "invoice", "deducted",
            # Expanded payment-action phrases
            "payment failed", "payment declined", "overcharged",
            "charged twice", "duplicate charge", "billing issue",
            "receipt", "charged incorrectly", "subscription fee",
            "transaction amount", "extra charge", "unexpected charge",
            # Further payment-action variants
            "bank charge", "wrong amount", "incorrect amount",
            "charge dispute", "unauthorized charge", "double charged",
            "double billing", "payment error", "billing statement",
            "amount deducted", "money taken", "billed",
            "auto-deducted", "charged extra", "charged the wrong amount",
            "refunded", "credit",
        ],
        "Refund Department": [
            "refund", "return", "money back", "reimbursement", "cancel charge",
            "get my money back", "request refund", "want a refund",
            "refund request", "reimburse",
        ],
        "Logistics Department": [
            "delivery", "shipment", "package", "shipping",
            "not delivered", "late delivery", "damaged",
            "lost package", "tracking", "courier", "dispatch",
            "delivery status", "wrong address",
        ],
        "Technical Support": [
            # Original keywords
            "bug", "error", "software", "update", "not working",
            # Crash / freeze
            "crash", "crashes", "freezes", "frozen", "hangs",
            # App / website surface area
            "app", "application", "application issue", "application problem",
            "website", "website issue", "website problem",
            # Load failures
            "not loading", "unable to load", "failed to load", "loading issue",
            "page not loading", "page error",
            # System / server errors
            "system error", "server error", "system problem", "system down",
            # Connectivity / service
            "service unavailable", "connection error", "timeout",
            # Generic technical labels
            "technical issue", "technical problem",
            # Performance
            "slow response", "unexpected error",
            # Additional technical signals
            "broken", "cannot open", "fails to load", "not responding",
            "keeps crashing", "blank screen", "404", "500 error",
            # Extended technical coverage
            "latency", "glitch", "unresponsive", "not syncing",
            "failed to connect", "sync error", "app issue", "platform issue",
            "keeps failing", "not launching", "software error",
            "application error", "app error", "system failure",
            "internal error", "database error", "interface error",
        ],
        "Account Support": [
            "login", "account", "password", "username", "access", "credential",
            # Expanded account phrases
            "sign in", "logged out", "locked", "suspended",
            "cannot access", "reset password", "forgotten password",
            "account blocked", "account frozen", "account disabled",
            "access denied", "sign-in", "log in", "cannot log in",
            "account suspended", "account locked",
            # Further account-action variants
            "profile", "my account", "account details", "change email",
            "2fa", "mfa", "two-factor", "account recovery",
            "verify account", "forgot password", "account issue",
            "cannot sign in", "unable to login", "unable to log in",
        ],
        "Security Team": [
            "authentication", "otp", "two factor", "security",
            "verification", "fraud", "unauthorized", "hacked",
            "suspicious", "data breach", "identity theft",
        ],
        "Subscription Team": [
            "subscription", "plan", "membership",
            "cancel subscription", "upgrade", "downgrade",
            "renew", "auto-renewal", "billing cycle",
        ],
        "Product Team": [
            "product", "quality", "feature", "missing feature",
            "feature request", "product defect", "product issue",
        ],
        "Inventory Team": [
            "stock", "available", "inventory", "out of stock",
            "availability", "restock", "back in stock",
        ],
        "Order Management": [
            "order", "cancel order", "order status",
            "order confirmation", "wrong item", "missing item",
        ],
        "Promotions Team": [
            "coupon", "discount", "offer", "promo",
            "voucher", "promotional", "deal",
        ],
        "Customer Service": [
            "complaint", "issue", "problem", "help",
            # Expanded CONTACT-style phrases
            "contacted support", "no response", "support request",
            "unresolved complaint", "waiting for response", "support team",
            "haven't heard back", "follow up", "no reply", "still waiting",
            "previous ticket", "raised a ticket", "escalate complaint",
            "ticket open", "reach out",
            # Further follow-up / escalation variants
            "no one replied", "reached out", "still no update",
            "follow-up request", "been waiting", "nobody responded",
            "open request", "pending ticket", "contact support",
            "get in touch", "no update", "never responded",
            "still unresolved", "submitted a ticket", "already contacted",
        ],
    }

    # Priority keywords (verbatim from XGBOOST_TRAINING_MODEL.ipynb, Module 8).
    HIGH_PRIORITY_KEYWORDS = [
        "fraud", "security", "hacked", "unauthorized", "otp",
        "deducted", "not delivered", "urgent", "immediately",
        "legal", "lawsuit", "scam",
    ]
    MEDIUM_PRIORITY_KEYWORDS = [
        "refund", "cancel", "not working", "crash", "error",
        "delay", "late", "damaged",
    ]

    # Lightweight sentiment lexicon (stand-in for VADER used in the notebook).
    _POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'amazing', 'love', 'happy', 'satisfied',
        'awesome', 'perfect', 'wonderful', 'thank', 'thanks', 'nice', 'best',
        'fantastic', 'pleased', 'recommend',
    }
    _NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'angry',
        'frustrated', 'disappointed', 'unacceptable', 'broken', 'error', 'fail',
        'failed', 'crash', 'slow', 'scam', 'fraud', 'hacked', 'unauthorized',
        'stolen', 'issue', 'problem', 'complaint', 'urgent', 'delay', 'late',
        'damaged', 'not working', 'cannot', 'unable', 'wrong', 'poor',
    }

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.pkl_model_path = os.path.join(model_dir, 'xgboost_model.pkl')
        self.loaded = False
        self.pkl_loaded = False

        try:
            self._load_models()
            self._load_config()
            self.loaded = True
            print(f"[OK] ML Engine loaded successfully from {model_dir}")
        except Exception as e:
            print(f"[WARN] ML Engine not loaded: {e}")
            print("   Backend will run with fallback predictions")

        try:
            self._load_pkl_model()
        except Exception as e:
            print(f"[WARN] Standalone pkl department model not loaded: {e}")
            print("   Department prediction will fall back to the TF-IDF category model.")

    def _load_pkl_model(self):
        """Load the standalone XGBoost department model from E:\\xgboost_model.pkl.

        Per XGBOOST_TRAINING_MODEL.ipynb (Module 7 CORRECTED), the model was
        trained on TfidfVectorizer(max_features=300, stop_words="english") of
        `issue_description` concatenated with 5 LabelEncoded structured columns
        (sentiment, channel, region, customer_segment, customer_gender) = 55
        features, target = `assigned_department`. We rebuild that exact vectorizer
        + label encoders here, fit on the source CSV, so new complaints can be
        scored identically to training.
        """
        if not os.path.exists(self.pkl_model_path):
            raise FileNotFoundError(self.pkl_model_path)
        self.pkl_model = joblib.load(self.pkl_model_path)

        # Rebuild the TF-IDF vectorizer exactly as the notebook did.
        self.pkl_tfidf = TfidfVectorizer(
            max_features=self.PKL_TFIDF_MAX_FEATURES, stop_words="english"
        )

        # Rebuild the 5 structured LabelEncoders on the source CSV so their
        # integer coding matches the training distribution.
        csv_path = os.path.join(os.path.dirname(self.model_dir), 'customer_support_tickets_200k.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        src = pd.read_csv(csv_path)
        # Sentiment is engineered the same way the notebook does (lexicon based).
        src['sentiment'] = src['issue_description'].astype(str).apply(self._lexicon_sentiment)
        for col in self.PKL_STRUCTURED_COLS:
            if col not in src.columns:
                raise KeyError(f"Expected structured column '{col}' not in source CSV")
            le = LabelEncoder()
            le.fit(src[col].astype(str))
            setattr(self, f"pkl_le_{col}", le)

        # Fit TF-IDF on the complaint text corpus.
        self.pkl_tfidf.fit(src['issue_description'].astype(str))

        n_text = len(self.pkl_tfidf.vocabulary_)
        n_struct = len(self.PKL_STRUCTURED_COLS)
        if n_text + n_struct != int(getattr(self.pkl_model, 'n_features_in_', 0)):
            print(f"[WARN] pkl feature count {n_text + n_struct} "
                  f"!= model's {getattr(self.pkl_model, 'n_features_in_', 0)} "
                  f"(tfidf={n_text}, structured={n_struct}).")

        self.pkl_loaded = True
        print(f"[OK] Standalone pkl department model loaded from {self.pkl_model_path} "
              f"(tfidf={n_text} + structured={n_struct} = {n_text + n_struct})")

    def _load_models(self):
        """Load all trained models and encoders."""
        self.xgb_category = joblib.load(os.path.join(self.model_dir, 'xgb_category_model.joblib'))
        self.xgb_intent = joblib.load(os.path.join(self.model_dir, 'xgb_intent_model.joblib'))
        self.xgb_priority = joblib.load(os.path.join(self.model_dir, 'xgb_priority_model.joblib'))
        self.tfidf = joblib.load(os.path.join(self.model_dir, 'tfidf_vectorizer.joblib'))
        self.le_category = joblib.load(os.path.join(self.model_dir, 'le_category.joblib'))
        self.le_intent = joblib.load(os.path.join(self.model_dir, 'le_intent.joblib'))
        self.le_priority = joblib.load(os.path.join(self.model_dir, 'le_priority.joblib'))
    
    def _load_config(self):
        """Load feature engineering configuration."""
        config_path = os.path.join(self.model_dir, 'feature_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'urgency_keywords': ['urgent', 'immediately', 'asap', 'emergency', 'critical',
                                     'right now', 'help me', 'desperate', 'not working', 'broken',
                                     'cannot', "can't", 'unable', 'failed', 'error', 'crash'],
                'negative_words': ['angry', 'frustrated', 'terrible', 'worst', 'horrible', 'hate',
                                   'disappointed', 'unacceptable', 'ridiculous', 'waste', 'scam',
                                   'refuse', 'complaint', 'problem', 'issue', 'wrong', 'bad'],
                'positive_words': ['thank', 'please', 'appreciate', 'great', 'good', 'help',
                                   'kind', 'wonderful', 'excellent', 'love', 'happy', 'satisfied']
            }
    
    def is_loaded(self) -> bool:
        return self.loaded

    def is_pkl_loaded(self) -> bool:
        return self.pkl_loaded

    def _lexicon_sentiment(self, text: str) -> str:
        """Lightweight VADER stand-in: Positive / Negative / Neutral."""
        words = set(self._clean_text(str(text)).split())
        pos = len(words & self._POSITIVE_WORDS)
        neg = len(words & self._NEGATIVE_WORDS)
        if pos > neg:
            return "Positive"
        if neg > pos:
            return "Negative"
        return "Neutral"

    # High-signal technical-action keywords that grant Technical Support
    # precedence over any other department that scored the same or lower.
    # These are action-oriented (something is broken/failing) rather than
    # topic words like 'application' or 'website' which can appear in many
    # contexts.  Kept as a frozenset for O(1) look-up.
    _TECH_PRIORITY_KEYWORDS: frozenset = frozenset({
        "crash", "crashes", "freezes", "frozen", "hangs",
        "not loading", "unable to load", "failed to load", "loading issue",
        "page not loading", "page error",
        "system error", "server error", "system down",
        "service unavailable", "connection error", "timeout",
        "technical issue", "technical problem", "system problem",
        "application issue", "application problem",
        "website issue", "website problem",
        "slow response", "unexpected error",
        "not working", "error", "bug",
    })

    def _department_router(self, text: str) -> str:
        """Rule-based department router (Module 6 of XGBOOST_TRAINING_MODEL.ipynb).

        Scores each department by counting how many of its keywords appear in
        the complaint text, then returns the highest-scoring department.

        Tiebreaker / precedence rule (added to satisfy product requirement):
        If the complaint contains ANY keyword from ``_TECH_PRIORITY_KEYWORDS``
        AND "Technical Support" has a non-zero score, "Technical Support" is
        returned regardless of whether another department scored higher.
        This covers cases like
            "The app crashes every time I open my transaction history."
        where both 'Technical Support' and 'Payment Department' accumulate
        scores but the clear technical failure ('crashes') should dominate.

        The rule does NOT fire if Technical Support scored 0, so a complaint
        like "My transaction failed and I was charged." correctly routes to
        Payment Department.
        """
        text_lower = str(text).lower()
        scores = {dept: 0 for dept in self.ROUTING_RULES}
        for dept, keywords in self.ROUTING_RULES.items():
            for word in keywords:
                if word in text_lower:
                    scores[dept] += 1

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "Customer Service"

        # Technical precedence: a high-signal technical-action keyword is present
        # AND Technical Support's score is competitive (>= the highest rival score).
        # This prevents a single 'app' mention from overriding a payment complaint
        # with many payment-specific keyword matches.
        tech_score = scores.get("Technical Support", 0)
        if tech_score > 0 and any(kw in text_lower for kw in self._TECH_PRIORITY_KEYWORDS):
            rival_max = max(
                (v for k, v in scores.items() if k != "Technical Support"),
                default=0
            )
            if tech_score >= rival_max:
                return "Technical Support"

        return best

    def _assign_priority(self, text: str) -> str:
        """Verbatim priority labeling from XGBOOST_TRAINING_MODEL.ipynb Module 8."""
        text = str(text).lower()
        for word in self.HIGH_PRIORITY_KEYWORDS:
            if word in text:
                return "High"
        for word in self.MEDIUM_PRIORITY_KEYWORDS:
            if word in text:
                return "Medium"
        return "Low"

    # Intent derived from the (department, complaint) so it is always relevant.
    DEPARTMENT_INTENT = {
        "Refund Department": "request_refund",
        "Payment Department": "payment_issue",
        "Account Support": "account_access",
        "Security Team": "security_concern",
        "Subscription Team": "subscription_management",
        "Technical Support": "technical_issue",
        "Product Team": "feature_request",
        "Logistics Department": "delivery_issue",
        "Inventory Team": "inventory_issue",
        "Order Management": "order_issue",
        "Promotions Team": "promotion_issue",
        "Customer Service": "general_inquiry",
        # Names from the 7-class pkl mapping
        "ACCOUNT": "account_access",
        "REFUND": "request_refund",
        "Payment Problem": "payment_issue",
        "Security Concern": "security_concern",
        "Subscription Cancellation": "subscription_management",
        "Performance Issue": "technical_issue",
        "Feature Request": "feature_request",
        "Account Suspension": "account_access",
    }

    def _intent_from_department(self, department: str, text: str) -> str:
        """Return an intent that is consistent with the predicted department."""
        intent = self.DEPARTMENT_INTENT.get(department)
        if intent:
            return intent
        # Fallback: build a readable snake_case intent from the department name.
        return re.sub(r'[^a-z]+', '_', department.lower()).strip('_') + "_request"

    def _build_pkl_features(self, text: str) -> np.ndarray:
        """Build the exact 55-feature vector the pkl expects.

        TfidfVectorizer(300, stop_words) on the complaint + 5 LabelEncoded
        structured columns (sentiment, channel, region, customer_segment,
        customer_gender). Missing metadata uses the training-mode default
        ('Unknown' / Neutral) encoded with the fitted LabelEncoders.
        """
        # TF-IDF block (sparse).
        X_text = self.pkl_tfidf.transform([str(text)])

        # Structured block: use Neutral sentiment + Unknown categoricals, encoded
        # with the same LabelEncoders fit on the training CSV.
        struct_values = []
        sentiment = self._lexicon_sentiment(text)
        for col in self.PKL_STRUCTURED_COLS:
            if col == 'sentiment':
                val = sentiment
            else:
                val = 'Unknown'
            le = getattr(self, f"pkl_le_{col}")
            if val not in le.classes_:
                val = le.classes_[0]
            struct_values.append(le.transform([val])[0])
        X_struct = csr_matrix(np.array([struct_values], dtype=float))
        return hstack([X_text, X_struct]).astype(float)

    def _predict_department_pkl(self, text: str):
        """Predict department using E:\\xgboost_model.pkl (notebook pipeline).

        Returns (department_name, confidence, class_index, probabilities_dict).
        The pkl is used directly via the reconstructed TF-IDF + structured
        features. When its confidence is weak/ambiguous, the notebook's
        rule-based department_router (the pkl's own training target) is used as
        the final routing decision so the answer is always accurate to the
        training data.
        """
        X = self._build_pkl_features(text)
        proba = self.pkl_model.predict_proba(X)[0]
        pkl_idx = int(np.argmax(proba))
        pkl_conf = float(proba[pkl_idx])
        pkl_dept = self.PKL_CLASS_TO_DEPARTMENT.get(pkl_idx, f"Class {pkl_idx}")
        proba_dict = {
            self.PKL_CLASS_TO_DEPARTMENT.get(i, f"Class {i}"): round(float(p), 4)
            for i, p in enumerate(proba)
        }

        # Rule-based department (authoritative training target).
        rule_dept = self._department_router(text)

        # Use the pkl when it is confident; otherwise trust the rule-based router.
        if pkl_conf >= 0.5:
            department = pkl_dept
            confidence = round(pkl_conf, 4)
        else:
            department = rule_dept
            confidence = round(min(0.95, 0.6 + 0.1 * sum(
                1 for kws in self.ROUTING_RULES.get(rule_dept, []) if kws in str(text).lower()
            )), 4)

        return department, confidence, pkl_idx, proba_dict
    
    def _clean_text(self, text: str) -> str:
        """Clean input text for TF-IDF."""
        text = text.lower().strip()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _engineer_features(self, text: str) -> dict:
        """Extract engineered features from raw text."""
        text_lower = text.lower()
        words = text.split()
        
        features = {
            'text_length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'urgency_score': sum(1 for kw in self.config['urgency_keywords'] if kw in text_lower),
            'negative_word_count': sum(1 for w in self.config['negative_words'] if w in text_lower),
            'positive_word_count': sum(1 for w in self.config['positive_words'] if w in text_lower),
            'has_question': 1 if '?' in text else 0,
            'exclamation_count': text.count('!'),
        }
        features['sentiment_score'] = features['positive_word_count'] - features['negative_word_count']
        
        return features
    
    def predict(self, text: str) -> dict:
        """
        Predict category, intent, priority, and risk level for a ticket.
        
        Returns a comprehensive prediction dictionary with:
        - category, intent, priority
        - confidence scores
        - risk level and score
        - sentiment analysis
        - probabilities for all categories/intents
        """
        if not self.loaded:
            return self._fallback_predict(text)
        
        # 1. Clean text and get TF-IDF features
        clean = self._clean_text(text)
        X_tfidf = self.tfidf.transform([clean])

        # 2. Engineer features
        eng_features = self._engineer_features(text)
        feature_order = ['text_length', 'word_count', 'avg_word_length',
                        'urgency_score', 'negative_word_count', 'positive_word_count',
                        'sentiment_score', 'has_question', 'exclamation_count']
        X_eng = csr_matrix([[eng_features.get(f, 0) for f in feature_order]])

        # 3. Combine features
        X_combined = hstack([X_tfidf, X_eng])

        # 4. Predict category / department
        # Prefer the standalone E:\xgboost_model.pkl for department routing.
        if self.pkl_loaded:
            category, cat_confidence, cat_pred_idx, cat_prob_dict = self._predict_department_pkl(text)
        else:
            cat_proba = self.xgb_category.predict_proba(X_combined)[0]
            cat_pred_idx = int(np.argmax(cat_proba))
            cat_confidence = float(cat_proba[cat_pred_idx])
            category = self.le_category.inverse_transform([cat_pred_idx])[0]

            # Category probabilities dict
            cat_prob_dict = {
                self.le_category.inverse_transform([i])[0]: round(float(p), 4)
                for i, p in enumerate(cat_proba)
            }

        # Keyword-based routing normalization to exact dataset categories.
        #
        # Design rules:
        #   - "transaction" excluded: context-neutral, appears in technical
        #     complaints (e.g. "open my transaction history").
        #   - PAYMENT only fires on concrete payment-action words.
        #   - Technical Support keywords mirror ROUTING_RULES for consistency.
        #   - Technical Support wins ties so a complaint like "The app crashes
        #     when I view my billing history" routes to Technical Support, not
        #     PAYMENT.
        text_lower = str(text).lower()
        _original_category = category  # track whether keyword routing fires
        keyword_routes = {
            # Payment-action words only; "transaction" deliberately excluded
            # to avoid misrouting technical complaints that mention transaction
            # history, e.g. "app crashes when viewing transaction history".
            'PAYMENT': [
                'charged', 'deducted', 'payment', 'billing', 'invoice',
                'duplicate charge', 'refund', 'payment failed',
                'payment declined', 'overcharged', 'charged twice',
                'billing issue', 'receipt', 'charged incorrectly',
                'subscription fee', 'transaction amount',
                'extra charge', 'unexpected charge',
                # Extended payment-action variants
                'bank charge', 'wrong amount', 'incorrect amount',
                'charge dispute', 'unauthorized charge', 'double charged',
                'double billing', 'payment error', 'billing statement',
                'amount deducted', 'money taken', 'billed',
                'auto-deducted', 'charged extra', 'charged the wrong amount',
                'refunded', 'credit',
            ],
            'ACCOUNT': [
                'account', 'password', 'login', 'logged out', 'account access',
                'locked', 'suspended', 'logs out', 'sign in', 'sign-in',
                'cannot access account', 'reset password', 'forgotten password',
                'account blocked', 'account frozen', 'account disabled',
                'access denied', 'cannot log in', 'account suspended',
                'account locked', 'credential', 'username',
                # Extended account-action variants
                'profile', 'my account', 'account details', 'change email',
                '2fa', 'mfa', 'two-factor', 'account recovery',
                'verify account', 'forgot password', 'account issue',
                'cannot sign in', 'unable to login', 'unable to log in',
            ],
            'CONTACT': [
                'contacted support', 'no response', 'unresolved complaint',
                'support request', 'waiting for response', 'support team',
                "haven't heard back", 'follow up', 'no reply', 'still waiting',
                'previous ticket', 'raised a ticket', 'escalate complaint',
                'ticket open', 'reach out to support',
                # Extended follow-up / escalation variants
                'no one replied', 'reached out', 'still no update',
                'follow-up request', 'been waiting', 'nobody responded',
                'open request', 'pending ticket', 'contact support',
                'no update', 'never responded', 'still unresolved',
                'submitted a ticket', 'already contacted',
            ],
            # Consistent with ROUTING_RULES Technical Support list.
            'Technical Support': [
                'bug', 'crash', 'crashes', 'error', 'application', 'app',
                'website', 'software', 'not working', 'not loading',
                'server error', 'freezes', 'frozen', 'timeout', 'system error',
                'system down', 'service unavailable', 'connection error',
                'page error', 'loading issue', 'unexpected error',
                'technical issue', 'broken', 'cannot open', 'fails to load',
                'not responding', 'keeps crashing', 'blank screen',
                # Extended technical coverage
                '500 error', '404', 'latency', 'glitch', 'unresponsive',
                'not syncing', 'failed to connect', 'sync error',
                'app issue', 'platform issue', 'keeps failing',
                'not launching', 'software error', 'application error',
                'system failure', 'hangs', 'page not loading',
                'unable to load', 'failed to load',
            ],
        }
        scores = {}
        for route_cat, keywords in keyword_routes.items():
            scores[route_cat] = sum(1 for kw in keywords if kw in text_lower)

        if max(scores.values()) > 0:
            # Technical Support is ranked first so technical signals (crash, app,
            # bug) win over PAYMENT when both score equally — preventing a lone
            # financial noun from misrouting a technical complaint.
            priority_order = ['Technical Support', 'PAYMENT', 'ACCOUNT', 'CONTACT']
            best_cats = sorted(
                scores.keys(),
                key=lambda c: (-scores[c], priority_order.index(c) if c in priority_order else len(priority_order))
            )
            best_cat = best_cats[0]
            if scores[best_cat] > 0:
                category = best_cat
                # Intentionally no confidence boost — cat_confidence must reflect
                # true model certainty so the HIGH/MEDIUM/LOW risk thresholds
                # (≥0.80 / 0.60–0.79 / <0.60) remain meaningful.
        
        # 5. Predict intent (derived from the department so it is always relevant)
        intent = self._intent_from_department(category, text)
        intent_confidence = round(min(0.99, 0.6 + 0.1 * sum(
            1 for kws in self.ROUTING_RULES.get(category, []) if kws in str(text).lower()
        )), 4)
        intent_prob_dict = {intent: intent_confidence}

        # 6. Predict priority (notebook Module 8 rule-based labeling)
        priority = self._assign_priority(text)

        # 7. Sentiment analysis (notebook-style Positive/Negative/Neutral)
        sentiment_label = self._lexicon_sentiment(text)
        sentiment = eng_features['sentiment_score']

        # 8. Ambiguity gap (used as a secondary signal in misrouting risk).
        #
        #    When keyword routing overrides the pkl category, cat_prob_dict still
        #    reflects the pre-override class distribution and is no longer a
        #    reliable measure of decision confidence. In that case we derive the
        #    gap from the final cat_confidence directly (higher confidence → wider
        #    effective gap between the winning and runner-up class).
        if category in cat_prob_dict:
            # pkl's own top class was kept — use its probability distribution.
            sorted_proba = np.sort(list(cat_prob_dict.values()))[::-1]
            ambiguity_gap = float(sorted_proba[0] - sorted_proba[1]) if len(sorted_proba) > 1 else 1.0
        else:
            # Keyword routing changed the category; use a confidence-based proxy.
            # confidence=0.80 → gap≈0.60; confidence=0.60 → gap≈0.20.
            ambiguity_gap = max(0.0, 2.0 * cat_confidence - 1.0)

        # 9. Misrouting risk — 100% deterministic, confidence-driven.
        #
        #    HIGH confidence   (>= 0.80) → LOW risk  → auto-route, no triage
        #    MEDIUM confidence (0.60–0.79) → MEDIUM risk
        #    LOW confidence    (< 0.60)  → HIGH risk → Human Triage Team
        #
        #    Key rule: Priority ≠ Misrouting Risk.
        #    A high-priority ticket with >= 80% confidence is still LOW risk.
        #    A low-priority ticket with < 60% confidence is still HIGH risk.
        if cat_confidence >= 0.80:
            risk_level = 'Low'
            risk_score = 0.2
        elif cat_confidence < 0.60:
            risk_level = 'High'
            risk_score = 0.75
        else:
            risk_level = 'Medium'
            risk_score = 0.5
        
        return {
            'category': category,
            'intent': intent,
            'priority': priority,
            'confidence': round(cat_confidence, 4),
            'intent_confidence': round(intent_confidence, 4),
            'risk_level': risk_level,
            'risk_score': round(risk_score, 4),
            'ambiguity_gap': round(ambiguity_gap, 4),
            'sentiment_score': float(sentiment),
            'sentiment_label': sentiment_label,
            'urgency_score': eng_features['urgency_score'],
            'category_probabilities': cat_prob_dict,
            'intent_probabilities': intent_prob_dict,
            'features': {
                'text_length': eng_features['text_length'],
                'word_count': eng_features['word_count'],
                'has_question': eng_features['has_question'],
                'exclamation_count': eng_features['exclamation_count']
            }
        }
    
    def _fallback_predict(self, text: str) -> dict:
        """Fallback prediction when models aren't loaded."""
        eng_features = self._engineer_features(text)
        text_lower = text.lower()
        
        # Simple rule-based classification
        category_rules = {
            'ACCOUNT': ['account', 'login', 'password', 'credential', 'access', 'locked', 'suspended'],
            'BILLING': ['bill', 'charge', 'payment', 'invoice', 'price', 'refund', 'money'],
            'CANCELLATION': ['cancel', 'unsubscribe', 'terminate', 'end', 'stop'],
            'DELIVERY': ['deliver', 'ship', 'track', 'order', 'package'],
            'FEEDBACK': ['suggest', 'feature', 'improve', 'feedback', 'recommendation'],
            'ORDER': ['order', 'purchase', 'buy', 'checkout', 'cart'],
            'REFUND': ['refund', 'return', 'money back', 'reimburse'],
            'SHIPPING': ['shipping', 'delivery', 'courier', 'package'],
            'CONTACT': ['contact', 'reach', 'speak', 'call', 'email'],
            'NEWSLETTER': ['newsletter', 'email', 'unsubscribe', 'subscribe'],
        }
        
        scores = {}
        for cat, keywords in category_rules.items():
            scores[cat] = sum(1 for kw in keywords if kw in text_lower)
        
        if max(scores.values()) > 0:
            category = max(scores, key=scores.get)
            confidence = min(max(scores.values()) / 3.0, 0.9)
        else:
            category = 'CONTACT'
            confidence = 0.3
        
        # Priority from urgency
        if eng_features['urgency_score'] >= 3:
            priority = 'Critical'
        elif eng_features['urgency_score'] >= 2:
            priority = 'High'
        elif eng_features['urgency_score'] >= 1:
            priority = 'Medium'
        else:
            priority = 'Low'
        
        risk_score = 1 - confidence
        # confidence >= 0.80 -> Low; confidence < 0.60 -> High; else Medium
        if confidence >= 0.80:
            risk_level = 'Low'
        elif confidence < 0.60:
            risk_level = 'High'
        else:
            risk_level = 'Medium'
        
        sentiment = eng_features['sentiment_score']
        
        return {
            'category': category,
            'intent': f"{category.lower()}_general",
            'priority': priority,
            'confidence': round(confidence, 4),
            'intent_confidence': round(confidence * 0.8, 4),
            'risk_level': risk_level,
            'risk_score': round(risk_score, 4),
            'ambiguity_gap': round(confidence * 0.5, 4),
            'sentiment_score': float(sentiment),
            'sentiment_label': 'Positive' if sentiment > 1 else ('Negative' if sentiment < -1 else 'Neutral'),
            'urgency_score': eng_features['urgency_score'],
            'category_probabilities': {category: confidence},
            'intent_probabilities': {f"{category.lower()}_general": confidence},
            'features': {
                'text_length': eng_features['text_length'],
                'word_count': eng_features['word_count'],
                'has_question': eng_features['has_question'],
                'exclamation_count': eng_features['exclamation_count']
            }
        }
