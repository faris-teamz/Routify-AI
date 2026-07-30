"""
ROUTIFY AI - FastAPI Backend
============================
Main application entry point with all routes.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import sqlite3
import json
import os
from datetime import datetime

from ml_engine import MLEngine

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="ROUTIFY AI API",
    description="AI-based Customer Support Ticket Misrouting Prediction System",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATABASE SETUP (SQLite - Temporary)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'ticketnow.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            customer_name TEXT DEFAULT 'Anonymous',
            customer_email TEXT DEFAULT '',
            channel TEXT DEFAULT 'Chatbot',
            
            -- AI Predictions
            predicted_category TEXT,
            predicted_intent TEXT,
            predicted_priority TEXT,
            confidence_score REAL,
            intent_confidence REAL,
            risk_level TEXT,
            risk_score REAL,
            sentiment_score REAL,
            
            -- All probabilities (JSON)
            category_probabilities TEXT,
            intent_probabilities TEXT,
            
            -- Status
            status TEXT DEFAULT 'Open',
            assigned_department TEXT,
            assigned_to TEXT,
            resolution_notes TEXT,
            is_triage INTEGER DEFAULT 0,
            
            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS triage_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id INTEGER NOT NULL,
            reason TEXT NOT NULL,
            original_prediction TEXT,
            assigned_to TEXT,
            action_taken TEXT,
            new_department TEXT,
            status TEXT DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            FOREIGN KEY (ticket_id) REFERENCES tickets(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id INTEGER,
            sender TEXT NOT NULL,
            message TEXT NOT NULL,
            message_type TEXT DEFAULT 'text',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticket_id) REFERENCES tickets(id)
        )
    ''')
    
    conn.commit()
    
    # Ensure required columns exist for existing tables
    try:
        cursor.execute("ALTER TABLE tickets ADD COLUMN misrouting_risk TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE tickets ADD COLUMN route TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE tickets ADD COLUMN requires_human INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE tickets ADD COLUMN resolution TEXT")
    except sqlite3.OperationalError:
        pass
    
    conn.close()

# Initialize DB on startup
init_db()

# ============================================================
# ML ENGINE
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'algorithm', 'trained_models')
ml_engine = MLEngine(MODEL_DIR)

# ============================================================
# PYDANTIC MODELS
# ============================================================
class TicketCreate(BaseModel):
    description: str = Field(..., min_length=5, max_length=2000)
    customer_name: str = Field(default="Anonymous")
    customer_email: str = Field(default="")
    channel: str = Field(default="Chatbot")

class TicketUpdate(BaseModel):
    status: Optional[str] = None
    assigned_department: Optional[str] = None
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolution: Optional[str] = None
    predicted_priority: Optional[str] = None

class TriageAction(BaseModel):
    action: str = Field(..., description="accept, reassign, escalate")
    new_department: Optional[str] = None
    notes: Optional[str] = None
    assigned_to: Optional[str] = None

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1)
    customer_name: str = Field(default="User")

# ============================================================
# PREDICTION ENDPOINT
# ============================================================
@app.post("/api/predict")
async def predict_ticket(ticket: TicketCreate):
    """Submit ticket text and get AI predictions."""
    try:
        prediction = ml_engine.predict(ticket.description)
        
        # Save to database
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tickets (
                description, customer_name, customer_email, channel,
                predicted_category, predicted_intent, predicted_priority,
                confidence_score, intent_confidence, risk_level, risk_score,
                sentiment_score, category_probabilities, intent_probabilities,
                status, assigned_department, is_triage,
                misrouting_risk, route, requires_human, resolution
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        ''', (
            ticket.description,
            ticket.customer_name,
            ticket.customer_email,
            ticket.channel,
            prediction['category'],
            prediction['intent'],
            prediction['priority'],
            prediction['confidence'],
            prediction['intent_confidence'],
            prediction['risk_level'],
            prediction['risk_score'],
            prediction['sentiment_score'],
            json.dumps(prediction['category_probabilities']),
            json.dumps(prediction['intent_probabilities']),
            'Triage' if prediction['risk_level'] == 'High' else 'Open',
            prediction['category'],
            1 if prediction['risk_level'] == 'High' else 0,
            prediction['risk_level'],
            prediction['category'],
            1 if prediction['risk_level'] == 'High' else 0
        ))
        
        ticket_id = cursor.lastrowid
        
        # If high risk, add to triage queue
        if prediction['risk_level'] == 'High':
            triage_reason = []
            if prediction['confidence'] < 0.60:
                triage_reason.append(f"Low confidence: {prediction['confidence']:.1%}")
            if prediction.get('ambiguity_gap', 1) < 0.15:
                triage_reason.append("Ambiguous: multiple categories likely")
            if prediction['sentiment_score'] < -1:
                triage_reason.append("Very negative sentiment detected")
            
            cursor.execute('''
                INSERT INTO triage_queue (ticket_id, reason, original_prediction, status)
                VALUES (?, ?, ?, 'Pending')
            ''', (
                ticket_id,
                ' | '.join(triage_reason) if triage_reason else 'High risk score',
                prediction['category']
            ))
        
        conn.commit()
        conn.close()
        
        return {
            "ticket_id": ticket_id,
            "prediction": prediction,
            "status": "triage" if prediction['risk_level'] == 'High' else "auto_routed",
            "message": f"Ticket #{ticket_id} created and " + (
                "sent to Triage Team for manual review" 
                if prediction['risk_level'] == 'High' 
                else f"auto-routed to {prediction['category']} department"
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# CHAT ENDPOINT (For chatbot)
# ============================================================
@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Handle chatbot conversation and ticket creation."""
    try:
        text = str(message.message).strip()
        text_lower = text.lower()
        
        is_greeting = (
            text_lower in {'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy', 'greetings'}
            or text_lower.startswith(('hi ', 'hello ', 'hey ', 'good morning ', 'good afternoon ', 'good evening '))
        )
        
        if is_greeting:
            return {
                "response": "Hi! 👋 Welcome to Routifyz. How can I help you today?"
            }
        
        # Run prediction on the message
        prediction = ml_engine.predict(message.message)
        
        # Save ticket
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tickets (
                description, customer_name, channel,
                predicted_category, predicted_intent, predicted_priority,
                confidence_score, intent_confidence, risk_level, risk_score,
                sentiment_score, category_probabilities, intent_probabilities,
                status, assigned_department, is_triage,
                misrouting_risk, route, requires_human, resolution
            ) VALUES (?, ?, 'Chatbot', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        ''', (
            message.message,
            message.customer_name,
            prediction['category'],
            prediction['intent'],
            prediction['priority'],
            prediction['confidence'],
            prediction['intent_confidence'],
            prediction['risk_level'],
            prediction['risk_score'],
            prediction['sentiment_score'],
            json.dumps(prediction['category_probabilities']),
            json.dumps(prediction['intent_probabilities']),
            'Triage' if prediction['risk_level'] == 'High' else 'Open',
            prediction['category'],
            1 if prediction['risk_level'] == 'High' else 0,
            prediction['risk_level'],
            prediction['category'],
            1 if prediction['risk_level'] == 'High' else 0
        ))
        
        ticket_id = cursor.lastrowid
        
        # If high risk, add to triage
        if prediction['risk_level'] == 'High':
            cursor.execute('''
                INSERT INTO triage_queue (ticket_id, reason, original_prediction, status)
                VALUES (?, ?, ?, 'Pending')
            ''', (ticket_id, 'High misrouting risk detected', prediction['category']))
        
        # Save chat messages
        cursor.execute('''
            INSERT INTO chat_messages (ticket_id, sender, message, message_type)
            VALUES (?, 'user', ?, 'text')
        ''', (ticket_id, message.message))
        
        sentiment_label = prediction.get('sentiment_label', 'Neutral')
        priority = prediction.get('priority', 'Low')
        
        if sentiment_label == 'Negative' or priority == 'High':
            response = (
                "I understand how frustrating this must be. "
                "I've created a support ticket for you, and our team will look into this right away."
            )
        elif sentiment_label == 'Positive':
            response = (
                "Thank you for reaching out! "
                "I've created a support ticket for you. Our team will review it shortly. "
                "Is there anything else I can help with?"
            )
        else:
            response = (
                "I've created a support ticket for you. "
                "Our support team will review it and take the necessary action."
            )
        
        cursor.execute('''
            INSERT INTO chat_messages (ticket_id, sender, message, message_type)
            VALUES (?, 'bot', ?, 'text')
        ''', (ticket_id, response))
        
        conn.commit()
        conn.close()
        
        return {
            "response": response,
            "ticket": {
                "ticket_id": ticket_id,
                "department": prediction['category'],
                "intent": prediction['intent'],
                "priority": prediction['priority'],
                "confidence": prediction['confidence'],
                "status": 'Triage' if prediction['risk_level'] == 'High' else 'Open',
                "route": prediction['category'],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# TICKET CRUD ENDPOINTS
# ============================================================
@app.get("/api/tickets")
async def get_tickets(
    status: Optional[str] = None,
    department: Optional[str] = None,
    priority: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get all tickets with optional filters."""
    conn = get_db()
    cursor = conn.cursor()
    
    query = "SELECT * FROM tickets WHERE 1=1"
    params = []
    
    if status:
        query += " AND status = ?"
        params.append(status)
    if department:
        query += " AND predicted_category = ?"
        params.append(department)
    if priority:
        query += " AND predicted_priority = ?"
        params.append(priority)
    if risk_level:
        query += " AND risk_level = ?"
        params.append(risk_level)
    
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    tickets = [dict(row) for row in cursor.fetchall()]
    
    # Get total count
    count_query = "SELECT COUNT(*) FROM tickets WHERE 1=1"
    count_params = []
    if status:
        count_query += " AND status = ?"
        count_params.append(status)
    if department:
        count_query += " AND predicted_category = ?"
        count_params.append(department)
    if priority:
        count_query += " AND predicted_priority = ?"
        count_params.append(priority)
    if risk_level:
        count_query += " AND risk_level = ?"
        count_params.append(risk_level)
    
    cursor.execute(count_query, count_params)
    total = cursor.fetchone()[0]
    
    conn.close()
    
    # Parse JSON fields
    for t in tickets:
        if t.get('category_probabilities'):
            t['category_probabilities'] = json.loads(t['category_probabilities'])
        if t.get('intent_probabilities'):
            t['intent_probabilities'] = json.loads(t['intent_probabilities'])
    
    return {"tickets": tickets, "total": total}

@app.get("/api/tickets/{ticket_id}")
async def get_ticket(ticket_id: int):
    """Get a specific ticket."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    ticket = cursor.fetchone()
    conn.close()
    
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    result = dict(ticket)
    if result.get('category_probabilities'):
        result['category_probabilities'] = json.loads(result['category_probabilities'])
    if result.get('intent_probabilities'):
        result['intent_probabilities'] = json.loads(result['intent_probabilities'])
    
    return result

@app.patch("/api/tickets/{ticket_id}")
async def update_ticket(ticket_id: int, update: TicketUpdate):
    """Update a ticket (reassign, change status, etc.)."""
    conn = get_db()
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if update.status:
        updates.append("status = ?")
        params.append(update.status)
        if update.status == 'Resolved':
            updates.append("resolved_at = ?")
            params.append(datetime.now().isoformat())
    if update.assigned_department:
        updates.append("assigned_department = ?")
        params.append(update.assigned_department)
    if update.assigned_to:
        updates.append("assigned_to = ?")
        params.append(update.assigned_to)
    if update.resolution_notes:
        updates.append("resolution_notes = ?")
        params.append(update.resolution_notes)
    if update.resolution:
        updates.append("resolution = ?")
        params.append(update.resolution)
    if update.predicted_priority:
        updates.append("predicted_priority = ?")
        params.append(update.predicted_priority)
    
    updates.append("updated_at = ?")
    params.append(datetime.now().isoformat())
    params.append(ticket_id)
    
    cursor.execute(f"UPDATE tickets SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()
    conn.close()
    
    return {"message": f"Ticket #{ticket_id} updated successfully"}

# ============================================================
# TRIAGE ENDPOINTS
# ============================================================
@app.get("/api/triage")
async def get_triage_queue():
    """Get all tickets in the triage queue."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT t.*, tq.id as triage_id, tq.reason as triage_reason, 
               tq.status as triage_status, tq.assigned_to as triage_assigned_to,
               tq.action_taken
        FROM triage_queue tq
        JOIN tickets t ON tq.ticket_id = t.id
        ORDER BY tq.created_at DESC
    ''')
    items = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    for item in items:
        if item.get('category_probabilities'):
            item['category_probabilities'] = json.loads(item['category_probabilities'])
        if item.get('intent_probabilities'):
            item['intent_probabilities'] = json.loads(item['intent_probabilities'])
    
    return {"triage_queue": items, "total": len(items)}

@app.post("/api/triage/{ticket_id}/action")
async def triage_action(ticket_id: int, action: TriageAction):
    """Take action on a triage ticket."""
    conn = get_db()
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    if action.action == "accept":
        # Accept the AI prediction
        cursor.execute('''
            UPDATE triage_queue SET status = 'Resolved', action_taken = 'Accepted AI prediction',
            resolved_at = ? WHERE ticket_id = ?
        ''', (now, ticket_id))
        cursor.execute('''
            UPDATE tickets SET status = 'Open', is_triage = 0, updated_at = ? WHERE id = ?
        ''', (now, ticket_id))
        
    elif action.action == "reassign":
        # Reassign to different department
        cursor.execute('''
            UPDATE triage_queue SET status = 'Resolved', action_taken = ?,
            new_department = ?, assigned_to = ?, resolved_at = ? WHERE ticket_id = ?
        ''', (f"Reassigned to {action.new_department}", action.new_department, 
              action.assigned_to, now, ticket_id))
        cursor.execute('''
            UPDATE tickets SET status = 'Open', is_triage = 0, 
            assigned_department = ?, assigned_to = ?, updated_at = ? WHERE id = ?
        ''', (action.new_department, action.assigned_to, now, ticket_id))
        
    elif action.action == "escalate":
        cursor.execute('''
            UPDATE triage_queue SET status = 'Escalated', action_taken = 'Escalated',
            assigned_to = ? WHERE ticket_id = ?
        ''', (action.assigned_to, ticket_id))
        cursor.execute('''
            UPDATE tickets SET status = 'Escalated', predicted_priority = 'Critical',
            updated_at = ? WHERE id = ?
        ''', (now, ticket_id))
    
    conn.commit()
    conn.close()
    
    return {"message": f"Triage action '{action.action}' completed for ticket #{ticket_id}"}

# ============================================================
# ANALYTICS ENDPOINTS
# ============================================================
@app.get("/api/analytics")
async def get_analytics():
    """Get dashboard analytics data."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Total tickets
    cursor.execute("SELECT COUNT(*) FROM tickets")
    total_tickets = cursor.fetchone()[0]
    
    # Status distribution
    cursor.execute("SELECT status, COUNT(*) as count FROM tickets GROUP BY status")
    status_dist = {row['status']: row['count'] for row in cursor.fetchall()}
    
    # Category distribution
    cursor.execute("SELECT predicted_category, COUNT(*) as count FROM tickets GROUP BY predicted_category ORDER BY count DESC")
    category_dist = {row['predicted_category']: row['count'] for row in cursor.fetchall()}
    
    # Priority distribution
    cursor.execute("SELECT predicted_priority, COUNT(*) as count FROM tickets GROUP BY predicted_priority ORDER BY count DESC")
    priority_dist = {row['predicted_priority']: row['count'] for row in cursor.fetchall()}
    
    # Risk distribution
    cursor.execute("SELECT risk_level, COUNT(*) as count FROM tickets GROUP BY risk_level")
    risk_dist = {row['risk_level']: row['count'] for row in cursor.fetchall()}
    
    # Average confidence
    cursor.execute("SELECT AVG(confidence_score) as avg_conf FROM tickets")
    avg_confidence = cursor.fetchone()['avg_conf'] or 0
    
    # Triage stats
    cursor.execute("SELECT COUNT(*) FROM triage_queue WHERE status = 'Pending'")
    pending_triage = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM triage_queue WHERE status = 'Resolved'")
    resolved_triage = cursor.fetchone()[0]
    
    # Misrouting rate (triage / total)
    misrouting_rate = (pending_triage + resolved_triage) / total_tickets * 100 if total_tickets > 0 else 0
    
    # Recent tickets
    cursor.execute("SELECT * FROM tickets ORDER BY created_at DESC LIMIT 10")
    recent_tickets = [dict(row) for row in cursor.fetchall()]
    
    # Channel distribution
    cursor.execute("SELECT channel, COUNT(*) as count FROM tickets GROUP BY channel ORDER BY count DESC")
    channel_dist = {row['channel']: row['count'] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        "total_tickets": total_tickets,
        "status_distribution": status_dist,
        "category_distribution": category_dist,
        "priority_distribution": priority_dist,
        "risk_distribution": risk_dist,
        "channel_distribution": channel_dist,
        "avg_confidence": round(avg_confidence, 4),
        "misrouting_rate": round(misrouting_rate, 2),
        "pending_triage": pending_triage,
        "resolved_triage": resolved_triage,
        "recent_tickets": recent_tickets
    }

@app.get("/api/analytics/model")
async def get_model_info():
    """Get ML model performance info."""
    report_path = os.path.join(MODEL_DIR, 'model_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            return json.load(f)
    return {"message": "Model report not available"}

# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "ml_engine": ml_engine.is_loaded(),
        "database": os.path.exists(DB_PATH),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# SEED DATA (for demo)
# ============================================================
@app.post("/api/seed")
async def seed_data():
    """Seed the database with sample tickets for demo purposes."""
    sample_tickets = [
        "I can't log into my account. It keeps showing 'invalid credentials' even though I'm using the right password.",
        "My payment was deducted but the order shows as failed. I need an immediate refund.",
        "The app is extremely slow and keeps crashing when I try to open the dashboard.",
        "I want to cancel my subscription. The service hasn't been working properly for weeks.",
        "There's a bug in the report generation feature. Numbers don't add up correctly.",
        "I received an unauthorized login alert. Someone might have access to my account.",
        "How do I change my billing address? I moved to a new location.",
        "The data sync between my phone and laptop isn't working. Updates don't reflect across devices.",
        "I'd like to request a feature for dark mode in the mobile app.",
        "My account was suspended without any explanation. I need this resolved urgently as I have a deadline.",
        "The checkout page throws an error when I try to use my credit card.",
        "I've been waiting for 3 days for a response to my previous ticket. This is unacceptable!",
        "Can you help me understand my billing statement? There are charges I don't recognize.",
        "The API keeps returning 500 errors. Our production system is down because of this.",
        "I want to upgrade my plan from Basic to Premium. What are the steps?",
    ]
    
    results = []
    for desc in sample_tickets:
        ticket = TicketCreate(description=desc, customer_name="Demo User", channel="Seed")
        result = await predict_ticket(ticket)
        results.append(result)
    
    return {"message": f"Seeded {len(results)} sample tickets", "tickets": results}

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
