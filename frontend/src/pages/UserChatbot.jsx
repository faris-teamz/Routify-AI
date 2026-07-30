import { useState, useEffect, useRef, useCallback } from 'react'
import '../App.css'

const API_BASE = 'http://localhost:8000/api'

const SIDEBAR_ITEMS = [
  { id: 'new', label: 'New Complaint', icon: '✦', section: 'support' },
  { id: 'tickets', label: 'My Tickets', icon: '◇', section: 'support' },
  { id: 'help', label: 'Help', icon: '?', section: 'other' },
  { id: 'settings', label: 'Settings', icon: '⚙', section: 'other' },
]

const PROCESSING_MESSAGES = [
  'Analyzing your complaint...',
  'Identifying the appropriate support team...',
  'Creating your support ticket...',
]

const MAX_CHARS = 2000

function renderMessageText(text) {
  return text.split('\n').map((line, i) => (
    <span key={i}>
      {line.split(/(\*\*[^*]+\*\*)/).map((part, j) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={j}>{part.slice(2, -2)}</strong>
        }
        return part
      })}
      {i < text.split('\n').length - 1 && <br />}
    </span>
  ))
}

function isGreeting(text) {
  const greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy', 'greetings']
  const cleaned = text.toLowerCase().trim()
  return greetings.some(g => cleaned === g || cleaned.startsWith(g + ' ') || cleaned.startsWith(g + ',') || cleaned.startsWith(g + '!'))
}

export default function UserChatbot() {
  const [theme, setTheme] = useState(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('routifyz-theme')
      if (saved === 'light' || saved === 'dark') return saved
    }
    return 'dark'
  })
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [activeSidebar, setActiveSidebar] = useState('new')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [processingStep, setProcessingStep] = useState(0)
  const [ticketResult, setTicketResult] = useState(null)
  const [error, setError] = useState(null)
  const [clearConfirm, setClearConfirm] = useState(false)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const processingIntervalRef = useRef(null)

  const isDark = theme === 'dark'

  useEffect(() => {
    localStorage.setItem('routifyz-theme', theme)
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          sender: 'bot',
          text: 'Hi! 👋 Tell me what happened, and I\'ll help you with it.',
          time: new Date().toISOString()
        }
      ])
    }
  }, [])

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(scrollToBottom, [messages])

  const startProcessing = useCallback(() => {
    let step = 0
    setProcessingStep(0)
    processingIntervalRef.current = setInterval(() => {
      step += 1
      setProcessingStep(step)
      if (step >= PROCESSING_MESSAGES.length) {
        clearInterval(processingIntervalRef.current)
      }
    }, 1200)
  }, [])

  const stopProcessing = useCallback(() => {
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current)
      processingIntervalRef.current = null
    }
    setProcessingStep(0)
  }, [])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || loading) return
    if (text.length > MAX_CHARS) return

    if (isGreeting(text)) {
      setTicketResult(null)
      setError(null)
      const userMsg = { sender: 'user', text, time: new Date().toISOString() }
      setMessages(prev => [...prev, userMsg])
      setInput('')
      setLoading(true)
      startProcessing()

      setTimeout(() => {
        stopProcessing()
        setMessages(prev => [...prev, {
          sender: 'bot',
          text: 'Hi! 👋 Welcome to Routifyz. How can I help you today?',
          time: new Date().toISOString()
        }])
        setLoading(false)
      }, 900)
      return
    }

    setTicketResult(null)
    setError(null)
    const userMsg = { sender: 'user', text, time: new Date().toISOString() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    startProcessing()

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, customer_name: 'User' })
      })
      const data = await res.json()
      stopProcessing()

      if (data.ticket) {
        setMessages(prev => [...prev, {
          sender: 'bot',
          text: data.response || 'Your ticket has been created successfully.',
          time: new Date().toISOString()
        }])
      } else {
        setMessages(prev => [...prev, {
          sender: 'bot',
          text: data.response || 'I processed your request.',
          time: new Date().toISOString()
        }])
      }
    } catch (e) {
      stopProcessing()
      setError('Unable to connect to the support system. Please make sure the backend is running.')
      setMessages(prev => [...prev, {
        sender: 'bot',
        text: 'Sorry, I\'m having trouble connecting to the support system right now. Please try again in a moment.',
        time: new Date().toISOString()
      }])
    }
    setLoading(false)
  }

  const handleClearChat = () => {
    setMessages([])
    setTicketResult(null)
    setError(null)
    setClearConfirm(false)
    setMessages([
      {
        sender: 'bot',
        text: 'Hi! 👋 Tell me what happened, and I\'ll help you with it.',
        time: new Date().toISOString()
      }
    ])
  }

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const closeTicketResult = () => {
    setTicketResult(null)
  }

  const groupedSidebar = SIDEBAR_ITEMS.reduce((acc, item) => {
    if (!acc[item.section]) acc[item.section] = []
    acc[item.section].push(item)
    return acc
  }, {})

  return (
    <div className={`routifyz-app ${isDark ? 'routifyz-dark' : 'routifyz-light'}`}>
      <header className="routifyz-header">
        <div className="routifyz-header-left">
          <button
            className="routifyz-sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? '◀' : '▶'}
          </button>
          <div className="routifyz-brand">
            <div className="routifyz-logo">
              <svg width="30" height="30" viewBox="0 0 28 28" fill="none">
                <rect width="28" height="28" rx="7" fill="url(#logo-grad)"/>
                <path d="M8 14L12 10L16 14L20 10" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M8 18L12 14L16 18L20 14" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.5"/>
                <defs>
                  <linearGradient id="logo-grad" x1="0" y1="0" x2="28" y2="28">
                    <stop stopColor="#22d3ee"/>
                    <stop offset="1" stopColor="#6366f1"/>
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <div className="routifyz-brand-text">
              <span className="routifyz-brand-name">ROUTIFY</span>

            </div>
          </div>
        </div>
        <div className="routifyz-header-right">
          <span className="routifyz-online">
            <span className="routifyz-online-dot"></span>
            Online
          </span>
          <button className="routifyz-theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
            {isDark ? '☀' : '☾'}
          </button>
        </div>
      </header>

      <div className="routifyz-body">
        {sidebarOpen && (
          <aside className="routifyz-sidebar mob-sidebar">
            {Object.entries(groupedSidebar).map(([section, items]) => (
              <div key={section} className="routifyz-sidebar-section">
                {section === 'support' ? 'Support' : 'More'}
                {items.map(item => (
                  <button
                    key={item.id}
                    className={`routifyz-sidebar-item ${activeSidebar === item.id ? 'active' : ''}`}
                    onClick={() => { setActiveSidebar(item.id); setMobileSidebarOpen(false); }}
                  >
                    <span className="routifyz-sidebar-icon">{item.icon}</span>
                    <span>{item.label}</span>
                  </button>
                ))}
              </div>
            ))}
          </aside>
        )}
        {mobileSidebarOpen && (
          <div className="routifyz-sidebar-overlay" onClick={() => { setMobileSidebarOpen(false); setSidebarOpen(false); }} />
        )}

        <main className="routifyz-main">
            <div className="routifyz-chat-layout">
              {messages.length === 1 && !loading && !ticketResult && (
                <div className="routifyz-welcome">
                  <div className="routifyz-welcome-icon">
                    <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                      <rect width="64" height="64" rx="16" fill="url(#welcome-grad)"/>
                      <path d="M32 16L40 28L32 24L24 28Z" fill="white" opacity="0.9"/>
                      <path d="M16 36L32 48L48 36V40C48 44 32 52 32 52C32 52 16 44 16 40Z" fill="white" opacity="0.7"/>
                      <defs>
                        <linearGradient id="welcome-grad" x1="0" y1="0" x2="64" y2="64">
                          <stop stopColor="#22d3ee"/>
                          <stop offset="1" stopColor="#6366f1"/>
                        </linearGradient>
                      </defs>
                    </svg>
                  </div>
                  <h1 className="routifyz-welcome-title">Welcome to Routifyz</h1>
                  <p className="routifyz-welcome-desc">
                    Hi! 👋 Tell me what happened, and I'll help you with it.
                  </p>
                </div>
              )}

              <div className="routifyz-messages">
                {messages.map((msg, i) => (
                  <div key={i} className={`routifyz-message ${msg.sender}`}>
                    <div className="routifyz-avatar">
                      {msg.sender === 'bot' ? (
                        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                          <circle cx="10" cy="10" r="9" fill="url(#av-grad)"/>
                          <path d="M10 4C10 4 6 7 6 11C6 13.2 7.8 15 10 15C12.2 15 14 13.2 14 11C14 7 10 4 10 4Z" fill="white" opacity="0.9"/>
                          <circle cx="8" cy="8" r="1.5" fill="#050a14"/>
                          <circle cx="12" cy="8" r="1.5" fill="#050a14"/>
                          <path d="M8 12Q10 14 12 12" stroke="#050a14" strokeWidth="1.2" fill="none"/>
                          <defs>
                            <linearGradient id="av-grad" x1="0" y1="0" x2="20" y2="20">
                              <stop stopColor="#22d3ee"/>
                              <stop offset="1" stopColor="#6366f1"/>
                            </linearGradient>
                          </defs>
                        </svg>
                      ) : (
                        <div className="routifyz-avatar-user">U</div>
                      )}
                    </div>
                    <div className="routifyz-bubble-wrapper">
                      <div className={`routifyz-bubble ${msg.sender === 'bot' ? 'routifyz-bot-bubble' : 'routifyz-user-bubble'}`}>
                        {renderMessageText(msg.text)}
                      </div>
                      <span className="routifyz-time">
                        {new Date(msg.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="routifyz-message bot">
                    <div className="routifyz-avatar">
                      <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                        <circle cx="10" cy="10" r="9" fill="url(#av-grad2)"/>
                        <path d="M10 4C10 4 6 7 6 11C6 13.2 7.8 15 10 15C12.2 15 14 13.2 14 11C14 7 10 4 10 4Z" fill="white" opacity="0.9"/>
                        <circle cx="8" cy="8" r="1.5" fill="#050a14"/>
                        <circle cx="12" cy="8" r="1.5" fill="#050a14"/>
                        <path d="M8 12Q10 14 12 12" stroke="#050a14" strokeWidth="1.2" fill="none"/>
                        <defs>
                          <linearGradient id="av-grad2" x1="0" y1="0" x2="20" y2="20">
                            <stop stopColor="#22d3ee"/>
                            <stop offset="1" stopColor="#6366f1"/>
                          </linearGradient>
                        </defs>
                      </svg>
                    </div>
                    <div className="routifyz-bubble-wrapper">
                      <div className="routifyz-bubble routifyz-bot-bubble">
                        <div className="routifyz-processing">
                          <div className="routifyz-processing-indicator">
                            <div className="routifyz-processing-dots">
                              <span></span><span></span><span></span>
                            </div>
                          </div>
                          <div className="routifyz-processing-content">
                            <span className="routifyz-processing-text">
                              {PROCESSING_MESSAGES[Math.min(processingStep, PROCESSING_MESSAGES.length - 1)]}
                            </span>
                            <div className="routifyz-processing-bar">
                              <div className="routifyz-processing-bar-fill"></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {error && (
                  <div className="routifyz-message bot">
                    <div className="routifyz-avatar">
                      <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                        <circle cx="10" cy="10" r="9" fill="url(#av-grad3)"/>
                        <text x="10" y="14" textAnchor="middle" fill="white" fontSize="10" fontWeight="700">!</text>
                        <defs>
                          <linearGradient id="av-grad3" x1="0" y1="0" x2="20" y2="20">
                            <stop stopColor="#f87171"/>
                            <stop offset="1" stopColor="#fbbf24"/>
                          </linearGradient>
                        </defs>
                      </svg>
                    </div>
                    <div className="routifyz-bubble-wrapper">
                      <div className="routifyz-bubble routifyz-error-bubble">
                        {error}
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              <div className="routifyz-input-area">
                <div className="routifyz-input-wrapper">
                  <textarea
                    ref={inputRef}
                    className="routifyz-input"
                    placeholder="Describe your issue..."
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={1}
                    disabled={loading}
                  />
                  <div className="routifyz-input-actions">
                    {messages.length > 1 && (
                      <button
                        className="routifyz-clear-btn"
                        onClick={() => setClearConfirm(true)}
                        disabled={loading}
                        title="Clear chat"
                      >
                        ✕
                      </button>
                    )}
                    <span className="routifyz-char-counter">
                      {input.length}/{MAX_CHARS}
                    </span>
                    <button
                      className="routifyz-send-btn"
                      onClick={handleSend}
                      disabled={!input.trim() || loading}
                    >
                      {loading ? (
                        <div className="routifyz-spinner" />
                      ) : (
                        <>
                          Send
                          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                            <path d="M1 1L15 8L1 15L7 8L1 1Z" fill="white"/>
                          </svg>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
        </main>
      </div>

      {clearConfirm && (
        <div className="routifyz-modal-overlay" onClick={() => setClearConfirm(false)}>
          <div className="routifyz-modal" onClick={e => e.stopPropagation()}>
            <h3 className="routifyz-modal-title">Clear Chat</h3>
            <p className="routifyz-modal-text">This will clear the current conversation. It will not delete any tickets from the database. Are you sure?</p>
            <div className="routifyz-modal-actions">
              <button className="routifyz-btn routifyz-btn-ghost" onClick={() => setClearConfirm(false)}>Cancel</button>
              <button className="routifyz-btn routifyz-btn-danger" onClick={handleClearChat}>Clear</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
