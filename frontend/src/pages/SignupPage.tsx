import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function SignupPage() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const doSignup = async () => {
    setMsg("");
    const e = email.trim();
    if (!e || !password) {
      setMsg("Enter email + password.");
      return;
    }
    if (password.length < 8) {
      setMsg("Password must be at least 8 characters.");
      return;
    }
    if (password !== confirm) {
      setMsg("Passwords do not match.");
      return;
    }
    setBusy(true);
    const res = await signup(e, password, displayName.trim());
    setBusy(false);
    if (!res.ok) {
      setMsg("Sign up failed: " + (res.error || ""));
      return;
    }
    navigate("/live");
  };

  return (
    <div className="page active" id="page-signup">
      <div className="login-shell">
        <div className="title">Woodchip Monitor</div>

        <div className="card login-card">
          <div className="login-title">Create your account</div>
          <div className="login-sub">
            Register to access live monitoring and events.
          </div>

          <div className="login-grid">
            <div className="login-info-container">
              <label className="hint">Name</label>
              <input
                type="text"
                placeholder="Your name"
                autoComplete="name"
                value={displayName}
                onChange={(ev) => setDisplayName(ev.target.value)}
              />
            </div>
            <div className="login-info-container">
              <label className="hint">Email</label>
              <input
                type="text"
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
                placeholder="At least 8 characters"
                autoComplete="new-password"
                value={password}
                onChange={(ev) => setPassword(ev.target.value)}
              />
            </div>
            <div className="login-info-container">
              <label className="hint">Confirm password</label>
              <input
                type="password"
                placeholder="••••••••"
                autoComplete="new-password"
                value={confirm}
                onChange={(ev) => setConfirm(ev.target.value)}
                onKeyDown={(ev) => {
                  if (ev.key === "Enter") doSignup();
                }}
              />
            </div>
          </div>

          <div className="login-actions">
            <button className="btn" disabled={busy} onClick={doSignup}>
              Sign up
            </button>
            <div className="login-err">{msg}</div>
          </div>

          <div className="login-sub" style={{ marginTop: 12 }}>
            Already have an account? <Link to="/login">Sign in</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
