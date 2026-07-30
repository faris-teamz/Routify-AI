import { useState } from 'react'

const ADMIN_CREDENTIALS = {
  username: 'admin',
  password: 'admin123'
}

export default function AdminLogin({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    if (username === ADMIN_CREDENTIALS.username && password === ADMIN_CREDENTIALS.password) {
      onLogin()
    } else {
      setError('Invalid username or password')
      setLoading(false)
    }
  }

  return (
    <div className="admin-login-page">
      <div className="admin-login-card">
        <div className="admin-login-brand">
          <div className="logo-icon">TN</div>
          <h2>Admin Portal</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>ROUTIFY AI</p>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="modal-field">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={e => { setUsername(e.target.value); setError('') }}
              placeholder="Enter username"
              autoFocus
            />
          </div>
          <div className="modal-field">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={e => { setPassword(e.target.value); setError('') }}
              placeholder="Enter password"
            />
          </div>
          {error && <div className="admin-login-error">{error}</div>}
          <button type="submit" className="admin-login-btn" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>
        <div className="admin-login-footer">
          <a href="/chatbot">← Back to User Portal</a>
        </div>
      </div>
    </div>
  )
}
