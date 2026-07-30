"""
Comprehensive smoke-test for the Routifyz department routing logic.
Covers: single-issue, dual-issue, complex, ambiguous/unseen complaints,
and validates that Priority != Misrouting Risk (the core triage rule).

Run from the backend/ directory:
    python test_tech_routing.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Patch __init__ so we don't need the actual model files on disk.
from unittest.mock import patch

with patch.object(__import__('ml_engine').MLEngine, '__init__', lambda self, *a, **kw: None):
    from ml_engine import MLEngine
    router = MLEngine.__new__(MLEngine)  # bypass __init__

# Attach the config that _engineer_features / _fallback_predict need.
router.config = {
    'urgency_keywords': [
        'urgent', 'immediately', 'asap', 'emergency', 'critical',
        'right now', 'help me', 'desperate', 'not working', 'broken',
        'cannot', "can't", 'unable', 'failed', 'error', 'crash',
    ],
    'negative_words': [
        'angry', 'frustrated', 'terrible', 'worst', 'horrible', 'hate',
        'disappointed', 'unacceptable', 'ridiculous', 'waste', 'scam',
        'refuse', 'complaint', 'problem', 'issue', 'wrong', 'bad',
    ],
    'positive_words': [
        'thank', 'please', 'appreciate', 'great', 'good', 'help',
        'kind', 'wonderful', 'excellent', 'love', 'happy', 'satisfied',
    ],
}

route = router._department_router

SEP = "=" * 60

# ==============================================================================
# SECTION 1 -- Single-Issue Routing  (_department_router)
# ==============================================================================
ROUTING_CASES = [

    # -- Technical Support: original keyword coverage --------------------------
    ("The application crashes every time I open my transaction history.",
     "Technical Support"),
    ("The app crashes when I try to log in.",
     "Technical Support"),
    ("Website is not loading at all.",
     "Technical Support"),
    ("Getting a server error on the checkout page.",
     "Technical Support"),
    ("System down - cannot place any order.",
     "Technical Support"),
    ("App freezes immediately after launch.",
     "Technical Support"),
    ("Page not loading after the recent payment update.",
     "Technical Support"),
    ("Failed to load my order history. Connection error.",
     "Technical Support"),
    ("Unexpected error while submitting my payment details.",
     "Technical Support"),
    ("Service unavailable when I try to view my billing.",
     "Technical Support"),
    ("Timeout error during checkout - is the server down?",
     "Technical Support"),
    ("The app hangs after I enter card details.",
     "Technical Support"),
    ("Slow response on the account settings page.",
     "Technical Support"),

    # -- Technical Support: extended/new keyword coverage ----------------------
    ("The app keeps crashing with a 500 error on every request.",
     "Technical Support"),
    ("There is a glitch in the interface - buttons are unresponsive.",
     "Technical Support"),
    ("Data is not syncing between my devices. Sync error every time.",
     "Technical Support"),
    ("Failed to connect to the server after the latest update.",
     "Technical Support"),
    ("App is not launching at all after the OS upgrade.",
     "Technical Support"),
    ("Experiencing high latency and the platform keeps failing.",
     "Technical Support"),

    # -- Payment Department: original + extended keywords ----------------------
    ("My transaction failed and I was charged twice.",
     "Payment Department"),
    ("I was billed the wrong amount on my invoice.",
     "Payment Department"),
    ("Card was deducted but order not placed.",
     "Payment Department"),
    ("There is a charge dispute on my credit card statement.",
     "Payment Department"),
    ("I was double charged for the same subscription.",
     "Payment Department"),
    ("Wrong amount was deducted from my bank account.",
     "Payment Department"),

    # -- Account Support: original + extended keywords -------------------------
    ("I cannot remember my password and cannot log in.",
     "Account Support"),
    ("My account has been locked.",
     "Account Support"),
    ("I forgot my password and cannot sign in to my profile.",
     "Account Support"),
    ("Need to change the email address on my account.",
     "Account Support"),
    ("My two-factor authentication is not working and I need account recovery.",
     "Account Support"),

    # -- Customer Service (CONTACT): extended follow-up keywords ---------------
    ("I contacted support three days ago and no one replied.",
     "Customer Service"),
    ("Already contacted your team but still no update on my complaint.",
     "Customer Service"),
    ("Submitted a ticket last week - nobody responded yet.",
     "Customer Service"),

    # -- Ambiguous: account keywords outscore lone 'system' --------------------
    # Account keywords ('account', 'access') accumulate more score than
    # a single 'system down' hit, so Account Support wins here.
    ("The system is down and I cannot access my account.",
     "Account Support"),

    # -- Other departments: must NOT misroute ----------------------------------
    ("I want to cancel my subscription.",
     "Subscription Team"),
    ("Package not delivered after two weeks.",
     "Logistics Department"),
]

# ==============================================================================
# SECTION 2 -- Dual-Issue & Complex Routing  (_department_router)
# Technical signals must dominate when the score is competitive.
# ==============================================================================
DUAL_ISSUE_CASES = [
    # tech crash + payment mention  -> Technical Support
    ("The app crashes every time I open my payment history.",
     "Technical Support"),
    # tech error + billing mention  -> Technical Support
    ("I keep getting a server error when I try to view my billing page.",
     "Technical Support"),
    # tech not-loading + strong account keywords ('account', 'access',
    # 'cannot access') -> Account Support wins because account score (3) > tech score (2).
    # This is correct: keyword scoring, not just tech-signal presence, decides the winner.
    ("The website is not loading and I cannot access my account settings.",
     "Account Support"),
    # tech freeze + subscription mention  -> Technical Support
    ("App freezes when I try to manage my subscription plan.",
     "Technical Support"),
    # payment clearly dominates, no strong tech signal
    ("I was charged twice and want a refund. No technical issue, just billing.",
     "Payment Department"),
    # account clearly dominates, no strong tech signal
    ("I forgot my password and my account is locked. Need to reset it.",
     "Account Support"),
    # contact follow-up with refund mention but no tech signal
    ("I contacted support a week ago about my refund. Still waiting, no reply.",
     "Customer Service"),
]

# ==============================================================================
# SECTION 3 -- Confidence -> Risk -> Triage  (_fallback_predict)
# Validates: Priority != Misrouting Risk
# Rule: >= 0.80 conf -> LOW risk | 0.60-0.79 -> MEDIUM | < 0.60 -> HIGH risk
# ==============================================================================
CONFIDENCE_RISK_CASES = [
    # Many ACCOUNT keyword matches -> high confidence -> LOW risk
    # 'urgent'/'immediately' -> High priority, but risk must remain LOW
    {
        "text": "My account has been locked. Need urgent access immediately.",
        "expected_risk": "Low",
        "note": "High priority (urgent/immediately) must NOT raise misrouting risk",
    },
    #    # PAYMENT via fallback: 2/3 keywords match -> conf=0.67 -> MEDIUM risk
    # (fallback is not the full ML path; MEDIUM is correct per the 3-tier rule)
    {
        "text": "I was charged twice for my subscription. Please issue a refund.",
        "expected_risk": "Medium",
        "note": "",
    },
    # Completely nonsensical input -> no keyword match -> zero confidence -> HIGH risk
    {
        "text": "zxqwerty blobfish 12345",
        "expected_risk": "High",
        "note": "Unseen/gibberish complaint must route to Human Triage Team",
    },
]


# ==============================================================================
# TEST RUNNER HELPERS
# ==============================================================================
passed = failed = 0


def run_routing(section_label, cases):
    global passed, failed
    print("\n" + SEP)
    print("  " + section_label)
    print(SEP)
    for complaint, expected in cases:
        result = route(complaint)
        if result == expected:
            passed += 1
            print("[PASS] -> " + result)
        else:
            failed += 1
            print("[FAIL] Expected '" + expected + "', got '" + result + "'")
            print("       Complaint: " + complaint)


# ==============================================================================
# RUN SECTION 1 & 2
# ==============================================================================
run_routing("SECTION 1 -- Single-Issue Routing", ROUTING_CASES)
run_routing("SECTION 2 -- Dual-Issue & Complex Routing", DUAL_ISSUE_CASES)

# ==============================================================================
# RUN SECTION 3 -- Confidence / Risk / Triage
# ==============================================================================
print("\n" + SEP)
print("  SECTION 3 -- Confidence -> Risk -> Triage (fallback predict)")
print(SEP)

for case in CONFIDENCE_RISK_CASES:
    text = case["text"]
    expected_risk = case["expected_risk"]
    result = router._fallback_predict(text)
    actual_risk = result["risk_level"]
    actual_priority = result["priority"]
    actual_conf = result["confidence"]

    if actual_risk == expected_risk:
        passed += 1
        status = "PASS"
    else:
        failed += 1
        status = "FAIL"

    conf_pct = str(round(actual_conf * 100)) + "%"
    line = ("[" + status + "] Conf=" + conf_pct +
            " | Priority=" + actual_priority +
            " | Risk=" + actual_risk +
            " (expected " + expected_risk + ")")
    if case.get("note"):
        line += "  [" + case["note"] + "]"
    print(line)
    print("       Text: " + text[:75] + ("..." if len(text) > 75 else ""))

# ==============================================================================
# SUMMARY
# ==============================================================================
total = passed + failed
print("\n" + SEP)
print("  RESULT: " + str(passed) + "/" + str(total) + " tests passed.")
print(SEP)
if failed:
    sys.exit(1)
