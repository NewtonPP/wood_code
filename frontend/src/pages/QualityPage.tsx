import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { hasPerm } from "../lib/rbac";
import { getRulesCurrent, getRuleVersions, rollbackRules, updateRules } from "../lib/api";
import type { RuleVersion, RulesConfig } from "../types";

const RULES_UPDATED_EVENT = "wcm:rules-updated";

export default function QualityPage() {
  const { user } = useAuth();
  const role = user?.role || "staff";
  const canEdit = hasPerm(role, "edit_rules");

  const [alarmThr, setAlarmThr] = useState("");
  const [refMm, setRefMm] = useState("");
  const [conf, setConf] = useState("");
  const [nms, setNms] = useState("");
  const [alarmEnabled, setAlarmEnabled] = useState(false);
  const [moistEnabled, setMoistEnabled] = useState(false);
  const [moistTopk, setMoistTopk] = useState("");
  const [moistEvery, setMoistEvery] = useState("");
  const [reason, setReason] = useState("");
  const [versions, setVersions] = useState<RuleVersion[]>([]);
  const [msg, setMsg] = useState<{ text: string; cls: string }>({ text: "", cls: "hint" });

  const loadQualityCurrent = async () => {
    if (!user) return;
    setMsg({ text: "", cls: "hint" });
    try {
      const data = await getRulesCurrent();
      const cfg = data.rules || {};
      setAlarmThr(cfg.alarm_threshold_mm != null ? Number(cfg.alarm_threshold_mm).toFixed(1) : "");
      setAlarmEnabled(!!cfg.alarm_enabled);
      setRefMm(cfg.ref_diam_mm != null ? Number(cfg.ref_diam_mm).toFixed(1) : "");
      setConf(cfg.conf_thr != null ? Number(cfg.conf_thr).toFixed(2) : "");
      setNms(cfg.nms_iou != null ? Number(cfg.nms_iou).toFixed(2) : "");
      setMoistEnabled(!!cfg.moisture_enabled);
      setMoistTopk(cfg.moisture_topk != null ? String(parseInt(String(cfg.moisture_topk), 10)) : "");
      setMoistEvery(cfg.moisture_every_n_frames != null ? String(parseInt(String(cfg.moisture_every_n_frames), 10)) : "");
      const src = data.source || "runtime";
      const v = data.version != null ? `v${data.version}` : "—";
      setMsg({ text: `Loaded current rules (${src}, ${v}).`, cls: "hint ok" });
    } catch (err) {
      setMsg({ text: "Failed: " + err, cls: "hint err" });
    }
  };

  useEffect(() => {
    loadQualityCurrent();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const apply = async () => {
    if (!user || !canEdit) return;
    const rules: RulesConfig = {
      alarm_threshold_mm: parseFloat(alarmThr),
      alarm_enabled: !!alarmEnabled,
      ref_diam_mm: refMm.trim() ? parseFloat(refMm) : (null as unknown as number),
      conf_thr: conf.trim() ? parseFloat(conf) : (null as unknown as number),
      nms_iou: nms.trim() ? parseFloat(nms) : (null as unknown as number),
      moisture_enabled: !!moistEnabled,
      moisture_topk: moistTopk.trim() ? parseInt(moistTopk, 10) : (null as unknown as number),
      moisture_every_n_frames: moistEvery.trim() ? parseInt(moistEvery, 10) : (null as unknown as number),
    };
    Object.keys(rules).forEach((k) => {
      const val = rules[k];
      if (val === null || (typeof val === "number" && Number.isNaN(val))) delete rules[k];
    });

    const r = reason.trim() || "ui_update";
    try {
      const res = await updateRules(rules, r);
      if (!res.ok) {
        setMsg({ text: "Apply failed: " + (await res.text()), cls: "hint err" });
        return;
      }
      const data = await res.json();
      setMsg({ text: `Applied new version v${data.version}.`, cls: "hint ok" });
      window.dispatchEvent(new Event(RULES_UPDATED_EVENT));
    } catch (err) {
      setMsg({ text: "Error: " + err, cls: "hint err" });
    }
  };

  const loadRuleVersions = async () => {
    if (!user) return;
    try {
      const data = await getRuleVersions(80);
      setVersions(data.versions || []);
      setMsg({ text: `Loaded ${(data.versions || []).length} versions.`, cls: "hint ok" });
    } catch (err) {
      setMsg({ text: "Versions failed: " + err, cls: "hint err" });
    }
  };

  const rollback = async (ver: number) => {
    try {
      const res = await rollbackRules(ver);
      if (!res.ok) {
        setMsg({ text: "Rollback failed: " + (await res.text()), cls: "hint err" });
        return;
      }
      const d2 = await res.json();
      setMsg({ text: `Rolled back to v${d2.rolled_back_to} (new applied v${d2.new_version}).`, cls: "hint ok" });
      await loadQualityCurrent();
      window.dispatchEvent(new Event(RULES_UPDATED_EVENT));
      await loadRuleVersions();
    } catch (err) {
      setMsg({ text: "Rollback error: " + err, cls: "hint err" });
    }
  };

  return (
    <div className="page active" id="page-quality">
      <div className="page-inner">
        <div className="row">
          <div className="card" style={{ flex: 1, minWidth: 320 }}>
            <div style={{ fontWeight: 800, marginBottom: 6 }}>Quality Rules</div>
            <div className="hint">Versioned configuration (thresholds + detector settings). Update requires permission.</div>

            <div className="row" style={{ marginTop: 10 }}>
              <div style={{ minWidth: 200, flex: 1 }}>
                <label className="hint">Alarm threshold (mm)</label>
                <input type="number" id="qr-alarm-thr" min={0} step={1} value={alarmThr} onChange={(e) => setAlarmThr(e.target.value)} />
              </div>
              <div style={{ minWidth: 200, flex: 1 }}>
                <label className="hint">Ref disk diameter (mm)</label>
                <input type="number" id="qr-ref-mm" min={1} step={0.5} value={refMm} onChange={(e) => setRefMm(e.target.value)} />
              </div>
              <div style={{ minWidth: 200, flex: 1 }}>
                <label className="hint">Conf thr (0–1)</label>
                <input type="number" id="qr-conf" min={0} max={1} step={0.05} value={conf} onChange={(e) => setConf(e.target.value)} />
              </div>
              <div style={{ minWidth: 200, flex: 1 }}>
                <label className="hint">NMS IoU (0–1)</label>
                <input type="number" id="qr-nms" min={0} max={1} step={0.05} value={nms} onChange={(e) => setNms(e.target.value)} />
              </div>
            </div>

            <div className="row">
              <div style={{ minWidth: 240 }}>
                <label className="hint">Alarm enabled</label>
                <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                  <input type="checkbox" id="qr-alarm-enabled" checked={alarmEnabled} onChange={(e) => setAlarmEnabled(e.target.checked)} />
                  Enabled
                </label>
              </div>

              <div style={{ minWidth: 240 }}>
                <label className="hint">Moisture enabled</label>
                <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                  <input type="checkbox" id="qr-moist-enabled" checked={moistEnabled} onChange={(e) => setMoistEnabled(e.target.checked)} />
                  Enabled
                </label>
              </div>

              <div style={{ minWidth: 180 }}>
                <label className="hint">Moisture top-K</label>
                <input type="number" id="qr-moist-topk" min={1} max={64} step={1} value={moistTopk} onChange={(e) => setMoistTopk(e.target.value)} />
              </div>

              <div style={{ minWidth: 200 }}>
                <label className="hint">Moisture every N</label>
                <input type="number" id="qr-moist-every" min={1} max={120} step={1} value={moistEvery} onChange={(e) => setMoistEvery(e.target.value)} />
              </div>
            </div>

            <div className="row" style={{ marginTop: 6 }}>
              <div style={{ flex: 1, minWidth: 260 }}>
                <label className="hint">Reason (audit trail)</label>
                <input type="text" id="qr-reason" placeholder="e.g., Tune oversize threshold for new batch" value={reason} onChange={(e) => setReason(e.target.value)} />
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
                <button className="btn" id="btn-qr-refresh" onClick={loadQualityCurrent}>
                  Reload
                </button>
                <button className="btn" id="btn-qr-apply" disabled={!canEdit} onClick={apply}>
                  Apply new version
                </button>
              </div>
            </div>

            <div className={msg.cls} id="qr-msg" style={{ marginTop: 8 }}>
              {msg.text}
            </div>
          </div>
        </div>

        <div className="row" style={{ gap: 12 }}>
          <div className="card" style={{ flex: 1, minWidth: 320 }}>
            <div style={{ fontWeight: 800, marginBottom: 6 }}>Versions</div>
            <div className="hint">Rollback creates a new applied version equal to the selected version.</div>
            <div className="row" style={{ marginTop: 8 }}>
              <button className="btn btn-ghost" id="btn-qr-load-versions" onClick={loadRuleVersions}>
                Load versions
              </button>
            </div>

            <div className="table-wrap" style={{ marginTop: 10, maxHeight: 360 }}>
              <table>
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>Created</th>
                    <th>By</th>
                    <th>Reason</th>
                    <th>Applied</th>
                    <th />
                  </tr>
                </thead>
                <tbody id="qr-versions-tbody">
                  {versions.map((v) => (
                    <tr key={v.version}>
                      <td>v{v.version}</td>
                      <td>{new Date(Date.parse(v.created_at)).toLocaleString()}</td>
                      <td>{v.created_by_email || "—"}</td>
                      <td>{(v.reason || "—").slice(0, 80)}</td>
                      <td>{v.applied ? <span className="tag green">YES</span> : <span className="tag">no</span>}</td>
                      <td>
                        <button className="btn btn-ghost" data-ver={v.version} onClick={() => rollback(v.version)}>
                          Rollback
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
