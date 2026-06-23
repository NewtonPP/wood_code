import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

interface Props {
  armed: boolean; // becomes true once logged in & routed to live
  openControls: () => void;
  closeControls: () => void;
  isControlsOpen: () => boolean;
}

interface Step {
  before?: () => Promise<void>;
  targetId: string;
  title: string;
  text: string;
  sub: string;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const TOUR_SESSION_KEY = "wcm_tour_seen";

function lockApp() {
  document.getElementById("app-shell")?.classList.add("app-locked");
}
function unlockApp() {
  document.getElementById("app-shell")?.classList.remove("app-locked");
}

export default function Tour({ armed, openControls, closeControls, isControlsOpen }: Props) {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [index, setIndex] = useState(0);

  const holeRef = useRef<HTMLDivElement>(null);
  const highlightRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const controlsBefore = useCallback(async () => {
    openControls();
    await sleep(180);
  }, [openControls]);

  const routeBefore = useCallback(
    (path: string, wait: number) => async () => {
      navigate(path);
      await sleep(wait);
    },
    [navigate]
  );

  // Steps mirror the original tourSteps array.
  const stepsRef = useRef<Step[]>([]);
  stepsRef.current = [
    { before: async () => { navigate("/live"); await sleep(120); }, targetId: "nav", title: "Navigation", text: "Use these tabs to switch between Live, Events, Quality Rules, Audit, and Devices.", sub: "Tabs may vary based on your role permissions." },
    { targetId: "nav-live", title: "Live Tab", text: "Live camera feed + overlay is here.", sub: "This is the default monitoring view." },
    { targetId: "panel-live", title: "Live View", text: "Detector overlay + reference disk. Oversize chips are highlighted.", sub: "Use Filters/Controls to tune thresholds." },
    { targetId: "panel-stats", title: "Statistics", text: "Rolling diameter stats and alarm status.", sub: "Useful for operators during a shift." },
    { targetId: "panel-hist", title: "Histogram", text: "Distribution of diameters with the threshold marker.", sub: "The oversize percentage is shown above the chart." },
    { targetId: "panel-moist", title: "Moisture", text: "Moisture model runs on top-K chip crops.", sub: "Enable/disable from Controls." },

    { before: controlsBefore, targetId: "btn-toggle-controls", title: "Filters / Controls", text: "Open this panel to adjust thresholds and model settings.", sub: "Some settings may be read-only depending on your role." },
    { before: controlsBefore, targetId: "group-theme", title: "Day/Night Mode", text: "Switch between light and dark themes.", sub: "Saved locally on this device." },
    { before: controlsBefore, targetId: "group-alarm-thr", title: "Alarm Threshold", text: "Set the oversize cutoff in mm.", sub: "This drives the red marker in the histogram." },
    { before: controlsBefore, targetId: "group-alarm-enable", title: "Enable Oversize Alarm", text: "Turn oversize highlighting on/off.", sub: "When enabled, oversize chips are emphasized in the live view." },
    { before: controlsBefore, targetId: "group-ref", title: "Reference Disk", text: "Optional: enter the real disk diameter for calibration.", sub: "Helps convert pixels to mm consistently." },
    { before: controlsBefore, targetId: "group-conf", title: "Confidence Threshold", text: "Filter out low-confidence detections.", sub: "Higher values = fewer boxes, lower values = more boxes." },
    { before: controlsBefore, targetId: "group-nms", title: "NMS IoU", text: "Controls how overlapping boxes are merged.", sub: "Lower values suppress overlaps more aggressively." },
    { before: controlsBefore, targetId: "group-moist-enable", title: "Moisture Inference", text: "Enable/disable the moisture model.", sub: "Runs on cropped chips from detections." },
    { before: controlsBefore, targetId: "group-moist-topk", title: "Moisture Top‑K", text: "How many chip crops to send to the moisture model.", sub: "Top‑K is based on detector confidence." },
    { before: controlsBefore, targetId: "group-moist-every", title: "Moisture Every N", text: "Run moisture every N processed frames.", sub: "Higher N reduces compute load." },
    { before: controlsBefore, targetId: "btn-save-config", title: "Apply", text: "Apply sends your control settings to the backend.", sub: "If you are read‑only, ask your admin for edit access." },

    { before: routeBefore("/events", 140), targetId: "nav-events", title: "Events Tab", text: "Browse historical telemetry snapshots.", sub: "Use filters and export for reporting." },
    { before: routeBefore("/events", 140), targetId: "btn-events-refresh", title: "Refresh Events", text: "Load the latest events based on the time range and filters.", sub: "Tip: “Oversize only” shows alarm events." },
    { before: routeBefore("/events", 140), targetId: "btn-events-export", title: "Export CSV", text: "Download events to CSV (if your role allows exports).", sub: "Great for QA audits and daily reports." },

    { before: routeBefore("/quality", 160), targetId: "nav-quality", title: "Quality Rules Tab", text: "Versioned configuration for thresholds and detector settings.", sub: "Shown only if you have permission." },
    { before: routeBefore("/quality", 160), targetId: "btn-qr-apply", title: "Apply New Version", text: "Creates a new version with your changes (with reason).", sub: "All changes are recorded in the Audit log." },
    { before: routeBefore("/quality", 160), targetId: "btn-qr-load-versions", title: "Versions", text: "View prior versions and rollback by re‑applying a version.", sub: "Rollback is saved as a new applied version." },

    { before: routeBefore("/audit", 160), targetId: "nav-audit", title: "Audit Tab", text: "See who did what and when (logins, exports, rules, errors).", sub: "Shown only if you have permission." },
    { before: routeBefore("/audit", 160), targetId: "btn-audit-refresh", title: "Refresh Audit", text: "Reload the latest audit entries.", sub: "Use this to confirm changes were recorded." },

    { before: routeBefore("/devices", 160), targetId: "nav-devices", title: "Devices Tab", text: "Health of the camera and inference loop.", sub: "Useful for debugging onsite." },
    { before: routeBefore("/devices", 160), targetId: "btn-dev-refresh", title: "Refresh Device Health", text: "Pull the latest device status from the backend.", sub: "Shows camera status, FPS, uptime, and last error." },
  ];

  const isInControls = (el: Element | null): boolean => {
    if (!el) return false;
    try {
      const panel = document.getElementById("controls-panel");
      const backdrop = document.getElementById("controls-backdrop");
      if (el === backdrop || el === panel) return true;
      const ids = ["btn-toggle-controls", "btn-close-controls", "btn-cancel-controls", "btn-save-config"];
      if (el.id && ids.includes(el.id)) return true;
      return !!(panel && panel.contains(el));
    } catch {
      return false;
    }
  };

  const isVisible = (el: Element | null): boolean =>
    !!(el && ((el as HTMLElement).offsetParent !== null || el === document.body));

  const placeSpotlight = useCallback((el: HTMLElement) => {
    const hole = holeRef.current;
    const highlight = highlightRef.current;
    const tooltip = tooltipRef.current;
    if (!hole || !highlight || !tooltip) return;

    const pad = 10;
    const r = el.getBoundingClientRect();
    const rr = {
      left: Math.max(6, r.left - pad),
      top: Math.max(6, r.top - pad),
      width: Math.min(window.innerWidth - 12, r.width + pad * 2),
      height: Math.min(window.innerHeight - 12, r.height + pad * 2),
    };

    hole.style.left = rr.left + "px";
    hole.style.top = rr.top + "px";
    hole.style.width = rr.width + "px";
    hole.style.height = rr.height + "px";

    highlight.style.left = rr.left + "px";
    highlight.style.top = rr.top + "px";
    highlight.style.width = rr.width + "px";
    highlight.style.height = rr.height + "px";

    const tooltipW = Math.min(420, Math.floor(window.innerWidth * 0.92));
    tooltip.style.width = tooltipW + "px";

    const rightSpace = window.innerWidth - (rr.left + rr.width);
    const leftSpace = rr.left;
    const belowSpace = window.innerHeight - (rr.top + rr.height);

    let x = 12,
      y = 12;
    if (rightSpace > tooltipW + 18) {
      x = rr.left + rr.width + 12;
      y = Math.max(12, Math.min(rr.top, window.innerHeight - 12 - tooltip.offsetHeight));
    } else if (leftSpace > tooltipW + 18) {
      x = rr.left - tooltipW - 12;
      y = Math.max(12, Math.min(rr.top, window.innerHeight - 12 - tooltip.offsetHeight));
    } else if (belowSpace > 140) {
      x = Math.max(12, Math.min(rr.left, window.innerWidth - 12 - tooltipW));
      y = rr.top + rr.height + 12;
    } else {
      x = Math.max(12, Math.min(rr.left, window.innerWidth - 12 - tooltipW));
      y = Math.max(12, rr.top - 12 - tooltip.offsetHeight);
    }
    tooltip.style.left = x + "px";
    tooltip.style.top = y + "px";
  }, []);

  const runStep = useCallback(
    async (target: number) => {
      const steps = stepsRef.current;
      const dir = target >= index ? 1 : -1;
      let i = Math.max(0, Math.min(steps.length - 1, target));

      while (i >= 0 && i < steps.length) {
        const cand = steps[i];
        if (cand && cand.before) {
          try {
            await cand.before();
          } catch {
            /* ignore */
          }
        }

        const tgt = document.getElementById(cand.targetId);
        if (isControlsOpen() && !isInControls(tgt)) {
          closeControls();
          await sleep(80);
        }

        const el2 = document.getElementById(cand.targetId);
        if (!el2 || isVisible(el2)) break;
        i += dir;
      }

      const clamped = Math.max(0, Math.min(steps.length - 1, i));
      setIndex(clamped);
      const el = document.getElementById(steps[clamped].targetId);
      requestAnimationFrame(() => {
        if (el) placeSpotlight(el as HTMLElement);
      });
    },
    [index, isControlsOpen, closeControls, placeSpotlight]
  );

  const openTour = useCallback(() => {
    setOpen(true);
    lockApp();
  }, []);
  const closeTour = useCallback(() => {
    setOpen(false);
    unlockApp();
  }, []);

  // start once per session, after armed
  useEffect(() => {
    if (!armed) return;
    if (sessionStorage.getItem(TOUR_SESSION_KEY)) return;
    sessionStorage.setItem(TOUR_SESSION_KEY, "1");
    openTour();
    runStep(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [armed]);

  // reposition on resize
  useEffect(() => {
    const onResize = () => {
      if (!open) return;
      const el = document.getElementById(stepsRef.current[index].targetId);
      if (el) placeSpotlight(el as HTMLElement);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [open, index, placeSpotlight]);

  const steps = stepsRef.current;
  const s = steps[index];

  return (
    <div className={"tour-layer" + (open ? " open" : "")} id="tour-layer" aria-hidden={!open}>
      <div className="tour-dim" />
      <div className="tour-hole" id="tour-hole" ref={holeRef} />
      <div className="tour-highlight" id="tour-highlight" ref={highlightRef} />

      <div className="tour-tooltip" id="tour-tooltip" role="dialog" aria-modal="true" ref={tooltipRef}>
        <div className="tour-title" id="tour-title">{s?.title || "Welcome"}</div>
        <div className="tour-text" id="tour-text">{s?.text || ""}</div>
        <div className="tour-sub" id="tour-sub">{s?.sub || ""}</div>

        <div className="tour-footer">
          <div className="tour-progress" id="tour-progress">
            Step {index + 1} of {steps.length}
          </div>
          <div className="tour-actions">
            <button className="btn btn-secondary" id="tour-skip" onClick={closeTour}>
              Skip
            </button>
            <button className="btn btn-secondary" id="tour-back" disabled={index === 0} onClick={() => runStep(index - 1)}>
              Back
            </button>
            <button
              className="btn"
              id="tour-next"
              onClick={() => {
                if (index === steps.length - 1) {
                  closeTour();
                  openControls();
                  return;
                }
                runStep(index + 1);
              }}
            >
              {index === steps.length - 1 ? "Finish" : "Next"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
