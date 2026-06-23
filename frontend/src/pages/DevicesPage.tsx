import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { getDeviceStatus } from "../lib/api";
import { fmtTs, fmtUptime } from "../lib/format";
import type { Health } from "../types";

export default function DevicesPage() {
  const { user } = useAuth();
  const [h, setH] = useState<Health | null>(null);
  const [msg, setMsg] = useState<{ text: string; cls: string }>({ text: "", cls: "hint" });

  const loadDeviceStatus = async () => {
    if (!user) return;
    setMsg({ text: "", cls: "hint" });
    try {
      const data = await getDeviceStatus();
      setH(data);
      setMsg({ text: "Updated.", cls: "hint ok" });
    } catch (err) {
      setMsg({ text: "Failed: " + err, cls: "hint err" });
    }
  };

  useEffect(() => {
    loadDeviceStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="page active" id="page-devices">
      <div className="page-inner">
        <div className="row">
          <div className="card" style={{ flex: 1, minWidth: 320 }}>
            <div style={{ fontWeight: 800, marginBottom: 6 }}>Device Health</div>
            <div className="hint">Status of camera + inference loop.</div>
            <div className="row" style={{ marginTop: 10 }}>
              <button className="btn" id="btn-dev-refresh" onClick={loadDeviceStatus}>
                Refresh
              </button>
              <div className={msg.cls} id="dev-msg">
                {msg.text}
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="row" style={{ gap: 16 }}>
            <div style={{ minWidth: 260 }}>
              <div className="hint">Camera</div>
              <div id="dev-camera" style={{ fontWeight: 800, color: h ? (h.camera_ok ? "var(--success)" : "var(--error)") : undefined }}>
                {h ? (h.camera_ok ? "OK • " + (h.cam_dev || "camera") : "OFFLINE") : "—"}
              </div>
            </div>
            <div style={{ minWidth: 260 }}>
              <div className="hint">Last frame</div>
              <div id="dev-last-frame" className="mono">
                {h ? fmtTs(h.last_frame_ts) : "—"}
              </div>
            </div>
            <div style={{ minWidth: 260 }}>
              <div className="hint">Last DETR infer</div>
              <div id="dev-last-detr" className="mono">
                {h ? fmtTs(h.last_detr_ts) : "—"}
              </div>
            </div>
            <div style={{ minWidth: 260 }}>
              <div className="hint">FPS (smoothed)</div>
              <div id="dev-fps" style={{ fontWeight: 800 }}>
                {h && h.fps_smoothed != null ? Number(h.fps_smoothed).toFixed(1) : "—"}
              </div>
            </div>
          </div>

          <div className="row" style={{ gap: 16, marginTop: 10 }}>
            <div style={{ minWidth: 260 }}>
              <div className="hint">Uptime</div>
              <div id="dev-uptime" style={{ fontWeight: 800 }}>
                {h ? fmtUptime(h.uptime_sec) : "—"}
              </div>
            </div>
            <div style={{ minWidth: 260 }}>
              <div className="hint">Alarm</div>
              <div id="dev-alarm" style={{ fontWeight: 800, color: h ? (h.alarm_active ? "var(--error)" : "var(--success)") : undefined }}>
                {h ? (h.alarm_active ? "OVERSIZE" : "OK") : "—"}
              </div>
            </div>
            <div style={{ minWidth: 260 }}>
              <div className="hint">Last error</div>
              <div id="dev-error" className="mono">
                {h ? h.last_error || "—" : "—"}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
