import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { hasPerm } from "../lib/rbac";
import { getEvents, eventsExportUrl } from "../lib/api";
import { fmtNum, parseLocalToEpoch } from "../lib/format";
import type { EventRow } from "../types";

export default function EventsPage() {
  const { user } = useAuth();
  const role = user?.role || "staff";
  const canExport = hasPerm(role, "export_events");

  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [alarmOnly, setAlarmOnly] = useState(false);
  const [limit, setLimit] = useState("200");
  const [rows, setRows] = useState<EventRow[]>([]);
  const [msg, setMsg] = useState<{ text: string; cls: string }>({ text: "", cls: "hint" });

  const buildQs = (defaultLimit: string): URLSearchParams => {
    const startEpoch = parseLocalToEpoch(start);
    const endEpoch = parseLocalToEpoch(end);
    const lim = parseInt(limit || defaultLimit, 10);
    const qs = new URLSearchParams();
    if (startEpoch != null) qs.set("start_epoch", String(startEpoch));
    if (endEpoch != null) qs.set("end_epoch", String(endEpoch));
    if (alarmOnly) qs.set("alarm_only", "true");
    if (Number.isFinite(lim)) qs.set("limit", String(lim));
    return qs;
  };

  const loadEvents = async () => {
    if (!user) return;
    setMsg({ text: "", cls: "hint" });
    try {
      const data = await getEvents(buildQs("200"));
      const evs = data.events || [];
      setRows(evs);
      setMsg({ text: `Loaded ${evs.length} events.`, cls: "hint ok" });
    } catch (err) {
      setMsg({ text: "Failed: " + err, cls: "hint err" });
    }
  };

  // auto-load on navigation (mirrors hashchange handler)
  useEffect(() => {
    loadEvents();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const exportCsv = () => {
    if (!user) return;
    window.location.href = eventsExportUrl(buildQs("10000"));
  };

  return (
    <div className="page active" id="page-events">
      <div className="page-inner">
        <div className="row">
          <div className="card" style={{ flex: 1, minWidth: 280 }}>
            <div style={{ fontWeight: 800, marginBottom: 6 }}>Events</div>
            <div className="hint">Browse history and export CSV for QA reporting.</div>

            <div className="row" style={{ marginTop: 10 }}>
              <div style={{ minWidth: 210, flex: 1 }}>
                <label className="hint">Start (local)</label>
                <input type="text" id="events-start" placeholder="YYYY-MM-DD HH:MM" value={start} onChange={(e) => setStart(e.target.value)} />
              </div>
              <div style={{ minWidth: 210, flex: 1 }}>
                <label className="hint">End (local)</label>
                <input type="text" id="events-end" placeholder="YYYY-MM-DD HH:MM" value={end} onChange={(e) => setEnd(e.target.value)} />
              </div>
              <div style={{ minWidth: 140 }}>
                <label className="hint">Alarm only</label>
                <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                  <input type="checkbox" id="events-alarm-only" checked={alarmOnly} onChange={(e) => setAlarmOnly(e.target.checked)} />
                  Oversize only
                </label>
              </div>
              <div style={{ minWidth: 140 }}>
                <label className="hint">Limit</label>
                <input type="number" id="events-limit" min={1} max={5000} step={50} value={limit} onChange={(e) => setLimit(e.target.value)} />
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
                <button className="btn" id="btn-events-refresh" onClick={loadEvents}>
                  Refresh
                </button>
                <button className="btn btn-ghost" id="btn-events-export" disabled={!canExport} onClick={exportCsv}>
                  Export CSV
                </button>
              </div>
            </div>

            <div className={msg.cls} id="events-msg" style={{ marginTop: 8 }}>
              {msg.text}
            </div>
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Alarm</th>
                <th>Max D (mm)</th>
                <th>Mean D</th>
                <th>Std D</th>
                <th>Units</th>
                <th>Moisture</th>
                <th>Device</th>
              </tr>
            </thead>
            <tbody id="events-tbody">
              {rows.map((e) => (
                <tr key={e.id}>
                  <td>{new Date(e.ts_epoch * 1000).toLocaleString()}</td>
                  <td>
                    {e.alarm_active ? (
                      <span className="tag red">OVERSIZE</span>
                    ) : (
                      <span className="tag green">OK</span>
                    )}
                  </td>
                  <td>{fmtNum(e.alarm_max_d_mm, 1)}</td>
                  <td>{fmtNum(e.mean_d, 1)}</td>
                  <td>{fmtNum(e.std_d, 1)}</td>
                  <td>{e.units || "—"}</td>
                  <td>{e.moisture_mean_pred || "—"}</td>
                  <td>{e.device_id || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
