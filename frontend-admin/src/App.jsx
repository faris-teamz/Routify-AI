import { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import AdminLogin from './pages/AdminLogin'
import AdminDashboard from './pages/AdminDashboard'

function App() {
  const [isAdminAuthenticated, setIsAdminAuthenticated] = useState(false)

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/admin" replace />} />
        <Route
          path="/admin"
          element={
            isAdminAuthenticated ?
              <AdminDashboard onLogout={() => setIsAdminAuthenticated(false)} /> :
              <AdminLogin onLogin={() => setIsAdminAuthenticated(true)} />
          }
        />
      </Routes>
    </BrowserRouter>
  )
}

export default App
