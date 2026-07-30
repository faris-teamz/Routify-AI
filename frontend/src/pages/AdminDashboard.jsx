import { useState, useEffect, useMemo } from 'react'

const API_BASE = 'https://routify-ai.onrender.com/api'

function Badge({ type, children }) {
  return <span className={`badge badge-${type}`}>{children}</span>
}

function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100)
  const color = pct >= 70 ? 'var(--success)' : pct >= 45 ? 'var(--warning)' : 'var(--danger)'
  return (
    <div className="confidence-bar">
      <div className="confidence-track">
        <div className="confidence-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="confidence-text" style={{ color }}>{pct}%</span>
    </div>
  )
}

function getRiskBadgeType(risk) {
  if (!risk) return 'risk-low'
  return `risk-${risk.toLowerCase()}`
}

function getPriorityBadgeType(priority) {
  if (!priority) return 'priority-low'
  return `priority-${priority.toLowerCase()}`
}

function getStatusBadgeType(status) {
  if (!status) return 'status'
  if (status === 'Triage') return 'status-triage'
  if (status === 'Resolved') return 'status-resolved'
  return 'status'
}

function formatDate(dateStr) {
  if (!dateStr) return '-'
  const d = new Date(dateStr)
  return d.toLocaleString('en-IN', { 
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit'
  })
}

function truncate(text, maxLen = 60) {
  if (!text) return '-'
  return text.length > maxLen ? text.slice(0, maxLen) + '...' : text
}

function Gauge({ value, label, max = 100, color = 'var(--accent-cyan)' }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  const rotation = (pct / 100) * 180
  return (
    <div className="admin-gauge-wrap">
      <div className="admin-gauge">
        <svg viewBox="0 0 120 70" className="admin-gauge-svg">
          <path d="M10 60 A50 50 0 0 1 110 60" fill="none" stroke="var(--bg-tertiary)" strokeWidth="10" strokeLinecap="round" />
          <path d="M10 60 A50 50 0 0 1 110 60" fill="none" stroke={color} strokeWidth="10" strokeLinecap="round" strokeDasharray={`${pct * 1.57} 157`} className="admin-gauge-fill" />
        </svg>
        <div className="admin-gauge-value">{Math.round(pct)}%</div>
      </div>
      <div className="admin-gauge-label">{label}</div>
    </div>
  )
}

function DonutChart({ segments, size = 100 }) {
  const total = segments.reduce((s, seg) => s + seg.value, 0) || 1
  let current = 0
  const gradient = segments.map(seg => {
    const start = (current / total) * 100
    current += seg.value
    const end = (current / total) * 100
    return `${seg.color} ${start}% ${end}%`
  }).join(', ')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <div className="admin-donut" style={{ width: size, height: size, background: `conic-gradient(${gradient})` }} />
      <div className="admin-legend">
        {segments.map((seg, i) => (
          <div key={i} className="admin-legend-item">
            <div className="admin-legend-dot" style={{ background: seg.color }} />
            <span>{seg.label} ({Math.round((seg.value / total) * 100)}%)</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function HorizontalBar({ segments }) {
  const total = segments.reduce((s, seg) => s + seg.value, 0) || 1
  return (
    <div style={{ width: '100%' }}>
      <div style={{ display: 'flex', height: 10, borderRadius: 5, overflow: 'hidden', gap: 2 }}>
        {segments.map((seg, i) => (
          <div key={i} style={{ width: `${(seg.value / total) * 100}%`, background: seg.color, borderRadius: 5, transition: 'all 0.6s cubic-bezier(0.4,0,0.2,1)' }} />
        ))}
      </div>
      <div className="admin-legend" style={{ marginTop: 10 }}>
        {segments.map((seg, i) => (
          <div key={i} className="admin-legend-item">
            <div className="admin-legend-dot" style={{ background: seg.color }} />
            <span>{seg.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function AdminDashboard({ onLogout }) {
  const [tickets, setTickets] = useState([])
  const [filters, setFilters] = useState({ status: '', department: '', priority: '', risk: '' })
  const [loading, setLoading] = useState(false)
  const [total, setTotal] = useState(0)
  const [selectedTicket, setSelectedTicket] = useState(null)

  useEffect(() => {
    fetchData()
  }, [filters])

  async function fetchData() {
    try {
      setLoading(true)
      const params = new URLSearchParams()
      if (filters.status) params.set('status', filters.status)
      if (filters.department) params.set('department', filters.department)
      if (filters.priority) params.set('priority', filters.priority)
      if (filters.risk) params.set('risk_level', filters.risk)
      
      const res = await fetch(`${API_BASE}/tickets?${params}&limit=100`)
      const data = await res.json()
      setTickets(data.tickets || [])
      setTotal(data.total || 0)
    } catch (e) {
      // Backend not available
    }
    setLoading(false)
  }

  async function updateTicketStatus(ticketId, status) {
    try {
      await fetch(`${API_BASE}/tickets/${ticketId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
      })
      fetchData()
    } catch (e) {
      // Error
    }
  }

  const categories = [...new Set(tickets.map(t => t.predicted_category).filter(Boolean))]

  const kpis = useMemo(() => {
    const totalTickets = tickets.length
    const openTickets = tickets.filter(t => t.status === 'Open').length
    const triageTickets = tickets.filter(t => t.requires_human || t.is_triage).length
    const highRiskTickets = tickets.filter(t => t.risk_level === 'High').length
    const avgConfidence = totalTickets > 0 
      ? Math.round((tickets.reduce((s, t) => s + (t.confidence_score || 0), 0) / totalTickets) * 100) 
      : 0
    
    let avgResolution = 0
    const resolvedTickets = tickets.filter(t => t.status === 'Resolved' && t.created_at && t.resolved_at)
    if (resolvedTickets.length > 0) {
      const totalMs = resolvedTickets.reduce((sum, t) => {
        const created = new Date(t.created_at).getTime()
        const resolved = new Date(t.resolved_at).getTime()
        return sum + (resolved - created)
      }, 0)
      avgResolution = Math.round(totalMs / resolvedTickets.length / (1000 * 60 * 60))
    }
    
    return { totalTickets, openTickets, triageTickets, highRiskTickets, avgConfidence, avgResolution }
  }, [tickets])

  const departmentStats = useMemo(() => {
    const stats = {}
    tickets.forEach(t => {
      const dept = t.predicted_category || 'Unknown'
      stats[dept] = (stats[dept] || 0) + 1
    })
    return Object.entries(stats)
      .map(([label, value]) => ({ label, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 6)
  }, [tickets])

  const statusStats = useMemo(() => {
    const stats = {}
    tickets.forEach(t => {
      const status = t.status || 'Unknown'
      stats[status] = (stats[status] || 0) + 1
    })
    return Object.entries(stats)
      .map(([label, value]) => ({ label, value }))
      .sort((a, b) => b.value - a.value)
  }, [tickets])

  const riskStats = useMemo(() => {
    const stats = { High: 0, Medium: 0, Low: 0 }
    tickets.forEach(t => {
      const risk = t.risk_level || 'Low'
      if (stats.hasOwnProperty(risk)) stats[risk]++
    })
    return Object.entries(stats).map(([label, value]) => ({ label, value }))
  }, [tickets])

  const deptColors = ['var(--accent-cyan)', 'var(--accent-royal)', 'var(--accent-purple)', 'var(--accent-teal)', 'var(--warning)', 'var(--info)']
  const statusColors = ['var(--success)', 'var(--warning)', 'var(--info)', 'var(--danger)', 'var(--accent-purple)']
  const riskColors = { High: 'var(--danger)', Medium: 'var(--warning)', Low: 'var(--success)' }

  const handleRowClick = (ticket) => {
    setSelectedTicket(ticket)
  }

  return (
    <div className="admin-portal">
      <header className="admin-header">
        <div className="admin-header-left">
          <div className="logo-icon">TN</div>
          <div>
            <h1 style={{ fontSize: '18px', fontWeight: 700 }}>ROUTIFY AI</h1>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Admin Dashboard</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{total} total complaints</span>
          <button onClick={onLogout} className="admin-logout-btn">Logout</button>
        </div>
      </header>

      {/* KPI Cards */}
      <div className="admin-kpi-section">
        <div className="admin-kpi-card">
          <div className="admin-kpi-label">Total Tickets</div>
          <div className="admin-kpi-value">{kpis.totalTickets}</div>
          <div className="admin-kpi-sub">All received complaints</div>
        </div>
        <div className="admin-kpi-card">
          <div className="admin-kpi-label">Open Tickets</div>
          <div className="admin-kpi-value">{kpis.openTickets}</div>
          <div className="admin-kpi-sub">Awaiting resolution</div>
        </div>
        <div className="admin-kpi-card">
          <div className="admin-kpi-label">High Risk</div>
          <div className="admin-kpi-value">{kpis.highRiskTickets}</div>
          <div className="admin-kpi-sub">Critical misrouting risk</div>
        </div>
        <div className="admin-kpi-card">
          <div className="admin-kpi-label">Human Triage</div>
          <div className="admin-kpi-value">{kpis.triageTickets}</div>
          <div className="admin-kpi-sub">Requires manual review</div>
        </div>
        <div className="admin-kpi-card">
          <div className="admin-kpi-label">Avg Confidence</div>
          <div className="admin-kpi-value">{kpis.avgConfidence}%</div>
          <div className="admin-kpi-sub">Model accuracy score</div>
        </div>
        <div className="admin-kpi-card">
          <div className="admin-kpi-label">Avg Resolution</div>
          <div className="admin-kpi-value">{kpis.avgResolution}h</div>
          <div className="admin-kpi-sub">Average time to resolve</div>
        </div>
      </div>

      {/* Filters */}
      <div className="admin-filters">
        <select className="admin-filter-select" value={filters.status} 
                onChange={e => setFilters(f => ({ ...f, status: e.target.value }))}>
          <option value="">All Status</option>
          <option value="Open">Open</option>
          <option value="Triage">Triage</option>
          <option value="In Progress">In Progress</option>
          <option value="Resolved">Resolved</option>
          <option value="Escalated">Escalated</option>
        </select>
        <select className="admin-filter-select" value={filters.priority}
                onChange={e => setFilters(f => ({ ...f, priority: e.target.value }))}>
          <option value="">All Priority</option>
          <option value="Critical">Critical</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>
        <select className="admin-filter-select" value={filters.risk}
                onChange={e => setFilters(f => ({ ...f, risk: e.target.value }))}>
          <option value="">All Risk</option>
          <option value="High">High Risk</option>
          <option value="Medium">Medium Risk</option>
          <option value="Low">Low Risk</option>
        </select>
        <select className="admin-filter-select" value={filters.department}
                onChange={e => setFilters(f => ({ ...f, department: e.target.value }))}>
          <option value="">All Departments</option>
          {categories.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <button className="admin-refresh-btn" onClick={fetchData}>
          {loading ? '...' : 'Refresh'}
        </button>
      </div>

      {/* Main Content Split */}
      <div className="admin-content">
        {/* Left Panel - Ticket Table */}
        <div className="admin-left-panel">
          <div className="admin-table-wrapper">
            {tickets.length === 0 ? (
              <div className="admin-empty">
                <div className="admin-empty-icon">&#x1F4CB;</div>
                <h3>No Complaints Yet</h3>
                <p>Complaints submitted via the chatbot will appear here.</p>
              </div>
            ) : (
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>Ticket ID</th>
                    <th>Complaint</th>
                    <th>Department</th>
                    <th>Priority</th>
                    <th>Confidence</th>
                    <th>Misrouting Risk</th>
                    <th>Route</th>
                    <th>Human Triage</th>
                    <th>Status</th>
                    <th>Resolution</th>
                    <th>Created</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {tickets.map(t => (
                    <tr 
                      key={t.id} 
                      className={selectedTicket && selectedTicket.id === t.id ? 'selected' : ''}
                      onClick={() => handleRowClick(t)}
                    >
                      <td className="admin-td-id">TN-{String(t.id).padStart(4, '0')}</td>
                      <td className="admin-td-desc" title={t.description}>{truncate(t.description, 50)}</td>
                      <td><Badge type="status">{t.predicted_category}</Badge></td>
                      <td>
                        <Badge type={getPriorityBadgeType(t.predicted_priority)}>
                          {t.predicted_priority}
                        </Badge>
                      </td>
                      <td><ConfidenceBar value={t.confidence_score || 0} /></td>
                      <td><Badge type={getRiskBadgeType(t.risk_level)}>{t.risk_level}</Badge></td>
                      <td style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{t.route || '-'}</td>
                      <td style={{ fontSize: '12px', textAlign: 'center' }}>
                        {(t.requires_human || t.is_triage) ? 'Yes' : 'No'}
                      </td>
                      <td><Badge type={getStatusBadgeType(t.status)}>{t.status}</Badge></td>
                      <td style={{ fontSize: '12px', color: 'var(--text-secondary)', maxWidth: '140px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={t.resolution || t.resolution_notes || '-'}>
                        {t.resolution || t.resolution_notes || '-'}
                      </td>
                      <td style={{ fontSize: '12px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                        {formatDate(t.created_at)}
                      </td>
                      <td>
                        {t.status !== 'Resolved' && (
                          <button 
                            className="admin-resolve-btn"
                            onClick={(e) => { e.stopPropagation(); updateTicketStatus(t.id, 'Resolved'); }}
                          >
                            Resolve
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Right Panel - Analytics Dashboard */}
        <div className="admin-right-panel">
          {!selectedTicket ? (
            <div className="admin-analytics-placeholder">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125v-11.25zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
              </svg>
              <h3>Select a complaint to view analytics</h3>
              <p>Click any row in the ticket table to see detailed analytics and insights for that complaint.</p>
            </div>
          ) : (
            <>
              {/* Ticket Summary */}
              <div className="admin-analytics-card">
                <div className="admin-analytics-title">Ticket Summary</div>
                <div className="admin-field-row">
                  <span className="admin-field-label">Complaint</span>
                  <span className="admin-field-value">{selectedTicket.description}</span>
                </div>
                <div className="admin-field-row">
                  <span className="admin-field-label">Department</span>
                  <span className="admin-field-value">{selectedTicket.predicted_category || '-'}</span>
                </div>
                <div className="admin-field-row">
                  <span className="admin-field-label">Priority</span>
                  <span className="admin-field-value">
                    <Badge type={getPriorityBadgeType(selectedTicket.predicted_priority)}>
                      {selectedTicket.predicted_priority}
                    </Badge>
                  </span>
                </div>
                <div className="admin-field-row">
                  <span className="admin-field-label">Confidence</span>
                  <span className="admin-field-value">
                    <ConfidenceBar value={selectedTicket.confidence_score || 0} />
                  </span>
                </div>
                <div className="admin-field-row">
                  <span className="admin-field-label">Misrouting Risk</span>
                  <span className="admin-field-value">
                    <Badge type={getRiskBadgeType(selectedTicket.risk_level)}>{selectedTicket.risk_level}</Badge>
                  </span>
                </div>
                <div className="admin-field-row">
                  <span className="admin-field-label">Human Triage</span>
                  <span className="admin-field-value">{(selectedTicket.requires_human || selectedTicket.is_triage) ? 'Yes' : 'No'}</span>
                </div>
                <div className="admin-field-row">
                  <span className="admin-field-label">Status</span>
                  <span className="admin-field-value">
                    <Badge type={getStatusBadgeType(selectedTicket.status)}>{selectedTicket.status}</Badge>
                  </span>
                </div>
              </div>

              {/* Confidence & Risk Gauges */}
              <div className="admin-analytics-card">
                <div className="admin-analytics-title">Confidence & Risk</div>
                <div className="admin-chart-row">
                  <div className="admin-chart-item">
                    <Gauge value={(selectedTicket.confidence_score || 0) * 100} label="Confidence" max={100} color="var(--success)" />
                  </div>
                  <div className="admin-chart-item">
                    <Gauge value={selectedTicket.risk_score ? selectedTicket.risk_score * 100 : 50} label="Risk Score" max={100} color="var(--danger)" />
                  </div>
                </div>
              </div>

              {/* Department Distribution */}
              <div className="admin-analytics-card">
                <div className="admin-analytics-title">Department Distribution</div>
                {departmentStats.length > 0 ? (
                  <HorizontalBar 
                    segments={departmentStats.map((s, i) => ({
                      ...s,
                      color: deptColors[i % deptColors.length]
                    }))}
                  />
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>No data available</div>
                )}
              </div>

              {/* Status Donut Chart */}
              <div className="admin-analytics-card">
                <div className="admin-analytics-title">Ticket Status</div>
                {statusStats.length > 0 ? (
                  <DonutChart 
                    size={120}
                    segments={statusStats.map((s, i) => ({
                      ...s,
                      color: statusColors[i % statusColors.length]
                    }))}
                  />
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>No data available</div>
                )}
              </div>

              {/* Risk Indicator */}
              <div className="admin-analytics-card">
                <div className="admin-analytics-title">Risk Indicator</div>
                {riskStats.length > 0 ? (
                  <HorizontalBar 
                    segments={riskStats.map(s => ({
                      ...s,
                      color: riskColors[s.label] || 'var(--text-muted)'
                    }))}
                  />
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>No data available</div>
                )}
              </div>

              {/* Resolution Progress */}
              <div className="admin-analytics-card">
                <div className="admin-analytics-title">Resolution Progress</div>
                <div style={{ marginBottom: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
                  {selectedTicket.status === 'Resolved' ? 'Completed' : 'In Progress'}
                </div>
                <div className="admin-progress-bar">
                  <div 
                    className="admin-progress-fill" 
                    style={{ width: selectedTicket.status === 'Resolved' ? '100%' : '45%' }} 
                  />
                </div>
              </div>

              {/* Timeline */}
              <div className="admin-analytics-card">
                <div className="admin-analytics-title">Timeline</div>
                <div className="admin-timeline">
                  <div className="admin-timeline-item">
                    <div className="admin-timeline-dot" />
                    <div className="admin-timeline-content">
                      <div className="admin-timeline-title">Ticket Created</div>
                      <div className="admin-timeline-time">{formatDate(selectedTicket.created_at)}</div>
                    </div>
                  </div>
                  {selectedTicket.updated_at && selectedTicket.updated_at !== selectedTicket.created_at && (
                    <div className="admin-timeline-item">
                      <div className="admin-timeline-dot" style={{ background: 'var(--accent-purple)' }} />
                      <div className="admin-timeline-content">
                        <div className="admin-timeline-title">Last Updated</div>
                        <div className="admin-timeline-time">{formatDate(selectedTicket.updated_at)}</div>
                      </div>
                    </div>
                  )}
                  {selectedTicket.resolved_at && (
                    <div className="admin-timeline-item">
                      <div className="admin-timeline-dot" style={{ background: 'var(--success)' }} />
                      <div className="admin-timeline-content">
                        <div className="admin-timeline-title">Resolved</div>
                        <div className="admin-timeline-time">{formatDate(selectedTicket.resolved_at)}</div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
