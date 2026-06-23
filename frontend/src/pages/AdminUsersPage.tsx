import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { createUser, listUsers, updateUser } from "../lib/api";
import type { AdminUser } from "../types";

const ROLES = ["staff", "quality_engineer", "software_engineer", "manager", "admin"];

export default function AdminUsersPage() {
  const { user } = useAuth();
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [msg, setMsg] = useState<{ text: string; cls: string }>({ text: "", cls: "hint" });

  // create-user form state
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("staff");
  const [busy, setBusy] = useState(false);

  const load = async () => {
    try {
      setUsers(await listUsers());
    } catch (err) {
      setMsg({ text: "Failed to load users: " + err, cls: "hint err" });
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const doCreate = async () => {
    setMsg({ text: "", cls: "hint" });
    const e = email.trim();
    if (!e || !password) {
      setMsg({ text: "Enter email + password.", cls: "hint err" });
      return;
    }
    setBusy(true);
    const res = await createUser(e, password, role, displayName.trim());
    setBusy(false);
    if (!res.ok) {
      let detail = "";
      try {
        detail = (await res.json())?.detail || "";
      } catch {
        detail = await res.text();
      }
      setMsg({ text: "Create failed: " + detail, cls: "hint err" });
      return;
    }
    setMsg({ text: `Created ${e} (${role}).`, cls: "hint ok" });
    setEmail("");
    setDisplayName("");
    setPassword("");
    setRole("staff");
    await load();
  };

  const changeRole = async (u: AdminUser, newRole: string) => {
    if (newRole === u.role) return;
    const res = await updateUser(u.id, { role: newRole });
    if (!res.ok) {
      setMsg({ text: "Update failed.", cls: "hint err" });
      return;
    }
    setMsg({ text: `${u.email} is now ${newRole}.`, cls: "hint ok" });
    await load();
  };

  const toggleActive = async (u: AdminUser) => {
    const next = u.is_active ? 0 : 1;
    const res = await updateUser(u.id, { is_active: !!next });
    if (!res.ok) {
      let detail = "";
      try {
        detail = (await res.json())?.detail || "";
      } catch {
        detail = await res.text();
      }
      setMsg({ text: "Update failed: " + detail, cls: "hint err" });
      return;
    }
    await load();
  };

  return (
    <div className="page active" id="page-admin-users">
      <div className="page-inner">
        <div className="row">
          <div className="card" style={{ flex: 1, minWidth: 320 }}>
            <div style={{ fontWeight: 800, marginBottom: 6 }}>Add a user</div>
            <div className="hint">
              Create an account and assign a role. Choose <b>admin</b> to grant full access.
            </div>
            <div className="row" style={{ marginTop: 10, flexWrap: "wrap", gap: 10 }}>
              <div style={{ minWidth: 180 }}>
                <label className="hint">Email</label>
                <input type="text" placeholder="name@company.com" value={email} onChange={(e) => setEmail(e.target.value)} />
              </div>
              <div style={{ minWidth: 160 }}>
                <label className="hint">Name</label>
                <input type="text" placeholder="Optional" value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
              </div>
              <div style={{ minWidth: 160 }}>
                <label className="hint">Password</label>
                <input type="password" placeholder="At least 8 chars" value={password} onChange={(e) => setPassword(e.target.value)} />
              </div>
              <div style={{ minWidth: 160 }}>
                <label className="hint">Role</label>
                <select value={role} onChange={(e) => setRole(e.target.value)}>
                  {ROLES.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </div>
              <button className="btn" disabled={busy} onClick={doCreate} style={{ alignSelf: "flex-end" }}>
                Create user
              </button>
              <div className={msg.cls}>{msg.text}</div>
            </div>
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Email</th>
                <th>Name</th>
                <th>Role</th>
                <th>Status</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => {
                const isSelf = u.email === user?.email;
                return (
                  <tr key={u.id}>
                    <td>{u.email}</td>
                    <td>{u.display_name || "—"}</td>
                    <td>
                      <select value={u.role} onChange={(e) => changeRole(u, e.target.value)}>
                        {ROLES.map((r) => (
                          <option key={r} value={r}>
                            {r}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td>{u.is_active ? "Active" : "Disabled"}</td>
                    <td>{new Date(Date.parse(u.created_at)).toLocaleString()}</td>
                    <td>
                      <button
                        className="btn btn-ghost"
                        disabled={isSelf}
                        title={isSelf ? "You cannot disable your own account" : ""}
                        onClick={() => toggleActive(u)}
                      >
                        {u.is_active ? "Disable" : "Enable"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
