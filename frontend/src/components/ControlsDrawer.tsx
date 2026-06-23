import { useCallback, useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import { hasPerm } from "../lib/rbac";
import { getRulesCurrent, updateRules } from "../lib/api";
import type { RulesConfig } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  reloadKey?: number; // bump to re-pull rules into the inputs
}

interface FormState {
  alarmThreshold: string;
  alarmEnabled: boolean;
  refSize: string;
  confThr: string;
  nmsIou: string;
  moistEnabled: boolean;
  moistTopk: string;
  moistEvery: string;
}

const EMPTY: FormState = {
  alarmThreshold: "",
  alarmEnabled: false,
  refSize: "",
  confThr: "",
  nmsIou: "",
  moistEnabled: false,
  moistTopk: "",
  moistEvery: "",
};

export default function ControlsDrawer({ open, onClose, reloadKey }: Props) {
  const { user } = useAuth();
  const { isLight, setLight } = useTheme();
  const [form, setForm] = useState<FormState>(EMPTY);
  const [hint, setHint] = useState<{ text: string; cls: string }>({ text: "", cls: "hint" });

  const role = user?.role || "staff";
  const canEdit = hasPerm(role, "edit_rules");

  useEffect(() => {
    setHint(
      canEdit
        ? { text: "You can apply changes (versioned via Quality Rules).", cls: "hint ok" }
        : { text: "Read-only: you don’t have permission to apply changes.", cls: "hint err" }
    );
  }, [canEdit]);

  // loadRuntimeRulesIntoControls
  const loadRules = useCallback(async () => {
    if (!user) return;
    try {
      const data = await getRulesCurrent();
      const cfg = data.rules || {};
      setForm((f) => ({
        ...f,
        alarmThreshold: cfg.alarm_threshold_mm != null ? Number(cfg.alarm_threshold_mm).toFixed(1) : f.alarmThreshold,
        alarmEnabled: !!cfg.alarm_enabled,
        confThr: cfg.conf_thr != null ? Number(cfg.conf_thr).toFixed(2) : f.confThr,
        nmsIou: cfg.nms_iou != null ? Number(cfg.nms_iou).toFixed(2) : f.nmsIou,
        refSize: cfg.ref_diam_mm != null ? Number(cfg.ref_diam_mm).toFixed(1) : f.refSize,
        moistEnabled: cfg.moisture_enabled != null ? !!cfg.moisture_enabled : f.moistEnabled,
        moistTopk: cfg.moisture_topk != null ? String(parseInt(String(cfg.moisture_topk), 10)) : f.moistTopk,
        moistEvery:
          cfg.moisture_every_n_frames != null ? String(parseInt(String(cfg.moisture_every_n_frames), 10)) : f.moistEvery,
      }));
    } catch {
      /* ignore */
    }
  }, [user]);

  useEffect(() => {
    loadRules();
  }, [loadRules, reloadKey]);

  // Re-pull when rules are applied/rolled back elsewhere (Quality page).
  useEffect(() => {
    const handler = () => loadRules();
    window.addEventListener("wcm:rules-updated", handler);
    return () => window.removeEventListener("wcm:rules-updated", handler);
  }, [loadRules]);

  const save = async () => {
    if (!user) return;
    if (!canEdit) {
      setHint({ text: "Not allowed: your role cannot apply changes.", cls: "hint err" });
      return;
    }
    const rules: RulesConfig = {
      alarm_threshold_mm: parseFloat(form.alarmThreshold),
      alarm_enabled: form.alarmEnabled,
      conf_thr: parseFloat(form.confThr),
      nms_iou: parseFloat(form.nmsIou),
      moisture_enabled: form.moistEnabled,
      moisture_topk: parseInt(form.moistTopk, 10),
      moisture_every_n_frames: parseInt(form.moistEvery, 10),
    };
    const refVal = parseFloat(form.refSize);
    if (!Number.isNaN(refVal)) rules.ref_diam_mm = refVal;

    try {
      const res = await updateRules(rules, "sidebar_controls_apply");
      if (!res.ok) {
        setHint({ text: "Apply failed: " + (await res.text()), cls: "hint err" });
        return;
      }
      setHint({ text: "Applied (new version created).", cls: "hint ok" });
    } catch (err) {
      setHint({ text: "Apply error: " + err, cls: "hint err" });
    }
  };

  return (
    <div
      className={"controls-panel-backdrop" + (open ? " open" : "")}
      id="controls-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className={"controls-panel" + (open ? " open" : "")} id="controls-panel">
        <div className="controls-header">
          <h2>Filters &amp; Controls</h2>
          <button className="btn btn-ghost" id="btn-close-controls" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="controls-body">
          <div className="controls-group" id="group-theme">
            <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input
                id="input-theme-light"
                type="checkbox"
                checked={isLight}
                onChange={(e) => setLight(e.target.checked)}
              />
              Day mode (light theme)
            </label>
            <small>Switch between night/day appearance. Saved on this device.</small>
          </div>

          <div className="controls-divider" />

          <div className="controls-group" id="group-alarm-thr">
            <label htmlFor="input-alarm-threshold">Alarm threshold (mm)</label>
            <input
              id="input-alarm-threshold"
              type="number"
              min={0}
              step={1}
              value={form.alarmThreshold}
              onChange={(e) => setForm({ ...form, alarmThreshold: e.target.value })}
            />
            <small>Chips above this size trigger the oversize alarm and are highlighted in red.</small>
          </div>

          <div className="controls-group" id="group-alarm-enable">
            <label>
              <input
                id="input-alarm-enabled"
                type="checkbox"
                checked={form.alarmEnabled}
                onChange={(e) => setForm({ ...form, alarmEnabled: e.target.checked })}
              />
              Enable oversize alarm
            </label>
            <small>When enabled, oversize detections are emphasized in the live view.</small>
          </div>

          <div className="controls-group" id="group-ref">
            <label htmlFor="input-ref-size">Reference disk diameter (mm)</label>
            <input
              id="input-ref-size"
              type="number"
              min={1}
              step={0.5}
              placeholder="e.g., 110"
              value={form.refSize}
              onChange={(e) => setForm({ ...form, refSize: e.target.value })}
            />
            <small>(Optional) Actual diameter of the reference disk used for calibration.</small>
          </div>

          <div className="controls-group" id="group-conf">
            <label htmlFor="input-conf-thr">Detection confidence threshold (0–1)</label>
            <input
              id="input-conf-thr"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={form.confThr}
              onChange={(e) => setForm({ ...form, confThr: e.target.value })}
            />
          </div>

          <div className="controls-group" id="group-nms">
            <label htmlFor="input-nms-iou">NMS IoU threshold (0–1)</label>
            <input
              id="input-nms-iou"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={form.nmsIou}
              onChange={(e) => setForm({ ...form, nmsIou: e.target.value })}
            />
          </div>

          <div className="controls-divider" />

          <div className="controls-group" id="group-moist-enable">
            <label>
              <input
                id="input-moist-enabled"
                type="checkbox"
                checked={form.moistEnabled}
                onChange={(e) => setForm({ ...form, moistEnabled: e.target.checked })}
              />
              Enable moisture inference
            </label>
            <small>Runs moisture model on chip crops (top-K by detector confidence).</small>
          </div>

          <div className="controls-group" id="group-moist-topk">
            <label htmlFor="input-moist-topk">Moisture top-K crops</label>
            <input
              id="input-moist-topk"
              type="number"
              min={1}
              max={64}
              step={1}
              value={form.moistTopk}
              onChange={(e) => setForm({ ...form, moistTopk: e.target.value })}
            />
          </div>

          <div className="controls-group" id="group-moist-every">
            <label htmlFor="input-moist-every">Moisture every N processed frames</label>
            <input
              id="input-moist-every"
              type="number"
              min={1}
              max={120}
              step={1}
              value={form.moistEvery}
              onChange={(e) => setForm({ ...form, moistEvery: e.target.value })}
            />
          </div>

          <div className="controls-divider" />

          <div className="controls-group">
            <div className={hint.cls} id="controls-perm-hint">
              {hint.text}
            </div>
          </div>
        </div>

        <div className="controls-footer">
          <button className="btn btn-ghost" id="btn-cancel-controls" onClick={onClose}>
            Close
          </button>
          <button className="btn" id="btn-save-config" disabled={!canEdit} onClick={save}>
            Apply
          </button>
        </div>
      </div>
    </div>
  );
}
