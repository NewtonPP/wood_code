import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { getAudit } from "../lib/api";
import type { AuditItem } from "../types";

export default function AuditPage() {
  const { user } = useAuth();
  const [limit, setLimit] = useState("200");
  const [items, setItems] = useState<AuditItem[]>([]);
  const [msg, setMsg] = useState<{ text: string; cls: string }>({ text: "", cls: "hint" });

  const loadAudit = async () => {
    if (!user) return;
    setMsg({ text: "", cls: "hint" });
    const lim = parseInt(limit || "200", 10);
    try {
      const data = await getAudit(lim);
      const its = data.items || [];
      setItems(its);
      setMsg({ text: `Loaded ${its.length} entries.`, cls: "hint ok" });
    } catch (err) {
      setMsg({ text: "Failed: " + err, cls: "hint err" });
    }
  };

  useEffect(() => {
    loadAudit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="page active" id="page-audit">
      <div className="page-inner">
        <div className="row">
          <div className="card" style={{ flex: 1, minWidth: 320 }}>
            <div style={{ fontWeight: 800, marginBottom: 6 }}>Audit Log</div>
            <div className="hint">Who did what, when (logins, rules changes, exports, errors).</div>
            <div className="row" style={{ marginTop: 10 }}>
              <div style={{ minWidth: 140 }}>
                <label className="hint">Limit</label>
                <input type="number" id="audit-limit" min={1} max={2000} step={50} value={limit} onChange={(e) => setLimit(e.target.value)} />
              </div>
              <button className="btn" id="btn-audit-refresh" onClick={loadAudit}>
                Refresh
              </button>
              <div className={msg.cls} id="audit-msg">
                {msg.text}
              </div>
            </div>
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>User</th>
                <th>Action</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody id="audit-tbody">
              {items.map((it) => {
                let details = "";
                try {
                  details = JSON.stringify(it.details || {});
                } catch {
                  details = "";
                }
                return (
                  <tr key={it.id}>
                    <td>{new Date(Date.parse(it.ts)).toLocaleString()}</td>
                    <td>{it.user_email || "—"}</td>
                    <td>{it.action || "—"}</td>
                    <td className="mono" style={{ maxWidth: 520 }}>
                      {details}
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
