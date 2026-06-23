import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const doLogin = async () => {
    setMsg("");
    const e = email.trim();
    if (!e || !password) {
      setMsg("Enter email + password.");
      return;
    }
    setBusy(true);
    const res = await login(e, password);
    setBusy(false);
    if (!res.ok) {
      setMsg("Login failed: " + (res.error || ""));
      return;
    }
    // after login: go live (or events if no view_live)
    // role comes from the freshly-set user; default to live which guards itself.
    navigate("/live");
  };

  return (
    <div className="page active" id="page-login">

      <div className="login-shell">
        <div className="title">
          Woodchip Monitor
        </div>
        
        <div className="card login-card">
          <div className="login-title">Welcome Back</div>
          <div className="login-sub">
            Use your assigned account to access monitoring, events, and quality controls.
          </div>

          <div className="login-grid">
            <div className="login-info-container">
              <label className="hint">Email</label>
              <input
                type="text"
                id="login-email"
                placeholder="name@company.com"
                autoComplete="username"
                value={email}
                onChange={(ev) => setEmail(ev.target.value)}
              />
            </div>
            <div className="login-info-container">
              <label className="hint">Password</label>
              <input
                type="password"
                id="login-password"
                placeholder="••••••••"
                autoComplete="current-password"
                value={password}
                onChange={(ev) => setPassword(ev.target.value)}
                onKeyDown={(ev) => {
                  if (ev.key === "Enter") doLogin();
                }}
              />
            </div>
          </div>

          <div className="login-actions">
            <button className="btn" id="btn-login" disabled={busy} onClick={doLogin}>
              Login
            </button>
            <div id="login-msg" className="login-err">
              {msg}
            </div>
          </div>

          <div className="login-sub" style={{ marginTop: 12 }}>
            Don't have an account? <Link to="/signup">Create one</Link>
          </div>
        </div>
      </div>
     </div>
  );
}
