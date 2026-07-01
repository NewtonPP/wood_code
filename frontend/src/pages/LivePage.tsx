import { useEffect, useState } from "react";
import { usePolling } from "../hooks/usePolling";
import { getHist, getMoisture, getDeviceInfo } from "../lib/api";
import Histogram from "../components/Histogram";
import MoistureBars from "../components/MoistureBars";
import LiveView from "../components/LiveView";
import type { HistData, Moisture, Stats } from "../types";

interface Props {
  stats: Stats | null;
}

function StatsPanel({ data }: { data: Stats | null }) {
  if (!data || !data.ready) {
    return (
      <>
        <div className="stat-line" id="stat-ready">
          <span className="stat-label">Status:</span>{" "}
          <span className="stat-value stat-warning">Collecting…</span>
        </div>
        <div className="stat-line" id="stat-units" />
        <div className="stat-line" id="stat-mean" />
        <div className="stat-line" id="stat-median" />
        <div className="stat-line" id="stat-std" />
        <div className="stat-line" id="stat-min" />
        <div className="stat-line" id="stat-max" />
        <div className="stat-line" id="stat-batch" />
        <div className="stat-line" id="stat-scale" />
        <div className="stat-line" id="stat-alarm" />
      </>
    );
  }

  const units = data.units || "";
  const mean = data.mean_d != null ? data.mean_d : data.mean;
  const med = data.median_d != null ? data.median_d : data.median;
  const std = data.std_d != null ? data.std_d : data.std;
  const minv = data.min_d != null ? data.min_d : data.min;
  const maxv = data.max_d != null ? data.max_d : data.max;

  return (
    <>
      <div className="stat-line" id="stat-ready">
        <span className="stat-label">Status:</span> <span className="stat-value">OK</span>
      </div>
      <div className="stat-line" id="stat-units">
        <span className="stat-label">Units:</span> <span className="stat-value">{units}</span>
      </div>
      <div className="stat-line" id="stat-mean">
        <span className="stat-label">Mean:</span> <span className="stat-value">{Number(mean).toFixed(1)} {units}</span>
      </div>
      <div className="stat-line" id="stat-median">
        <span className="stat-label">Median:</span> <span className="stat-value">{Number(med).toFixed(1)} {units}</span>
      </div>
      <div className="stat-line" id="stat-std">
        <span className="stat-label">Std:</span> <span className="stat-value">{Number(std).toFixed(1)} {units}</span>
      </div>
      <div className="stat-line" id="stat-min">
        <span className="stat-label">Min:</span> <span className="stat-value">{Number(minv).toFixed(1)} {units}</span>
      </div>
      <div className="stat-line" id="stat-max">
        <span className="stat-label">Max:</span> <span className="stat-value">{Number(maxv).toFixed(1)} {units}</span>
      </div>
      <div className="stat-line" id="stat-batch">
        {data.batch_label ? (
          <>
            <span className="stat-label">Batch:</span> <span className="stat-value">{data.batch_label}</span>
          </>
        ) : (
          ""
        )}
      </div>
      <div className="stat-line" id="stat-scale">
        {data.px_per_mm_mean != null ? (
          <>
            <span className="stat-label">px/mm:</span>{" "}
            <span className="stat-value">
              {Number(data.px_per_mm_mean).toFixed(1)} +/- {Number(data.px_per_mm_std || 0).toFixed(1)}
            </span>
          </>
        ) : (
          ""
        )}
      </div>
      <div className="stat-line" id="stat-alarm">
        {data.alarm_active ? (
          <>
            <span className="stat-label">Alarm:</span>{" "}
            <span className="stat-value stat-error">
              OVERSIZE &gt; {Number(data.alarm_threshold_mm != null ? data.alarm_threshold_mm : 0).toFixed(1)} mm (max{" "}
              {data.alarm_max_d_mm != null ? Number(data.alarm_max_d_mm).toFixed(1) : "?"})
            </span>
          </>
        ) : data.alarm_threshold_mm != null ? (
          <>
            <span className="stat-label">Alarm:</span>{" "}
            <span className="stat-value">OK (≤ {Number(data.alarm_threshold_mm).toFixed(1)} mm)</span>
          </>
        ) : (
          ""
        )}
      </div>
    </>
  );
}

export default function LivePage({ stats }: Props) {
  const [hist, setHist] = useState<HistData | null>(null);
  const [moist, setMoist] = useState<Moisture | null>(null);
  const [role, setRole] = useState<string>("device");

  useEffect(() => {
    getDeviceInfo().then((info) => setRole(info.role));
  }, []);

  usePolling(async () => {
    try {
      setHist(await getHist());
    } catch {
      /* ignore */
    }
  }, 3000);
  usePolling(async () => {
    try {
      setMoist(await getMoisture());
    } catch {
      setMoist((m) => ({ ...(m || {}), ready: false }));
    }
  }, 1200);

  // ---- histogram derived values (ported from updateHist) ----
  const histReady = !!(hist && hist.ready && hist.diameter && hist.diameter.bins && hist.diameter.counts);
  let overInfo = "";
  if (hist && hist.ready && hist.diameter && hist.pct_over_threshold != null) {
    const u = hist.units || "";
    if (u === "mm" && hist.alarm_threshold_mm != null) {
      overInfo = "Oversize (> " + Number(hist.alarm_threshold_mm).toFixed(0) + " mm): " + Number(hist.pct_over_threshold).toFixed(1) + "%";
    } else if (hist.threshold_x != null) {
      overInfo = "Oversize (> " + Number(hist.threshold_x).toFixed(0) + (u ? " " + u : "") + "): " + Number(hist.pct_over_threshold).toFixed(1) + "%";
    } else {
      overInfo = "Oversize: " + Number(hist.pct_over_threshold).toFixed(1) + "%";
    }
  }
  const histOpts = hist
    ? {
        thresholdX: hist.threshold_x,
        pctOver: hist.pct_over_threshold,
        units: hist.units || "",
        thrLabelVal: hist.units === "mm" && hist.alarm_threshold_mm != null ? hist.alarm_threshold_mm : hist.threshold_x,
      }
    : {};

  // ---- moisture derived (ported from updateMoisture) ----
  const mReady = !!(moist && moist.ready);
  const classes = (moist?.classes as string[]) || [];

  return (
    <div className="page active" id="page-live">
      <div className="top-row">
        {/* <div className="panel live-panel" id="panel-live"> */}
          {/* <h2>Live View</h2> */}
          <div className="live-content">
            <LiveView role={role} />
            <div className="live-legend">
              <div className="legend-row">
                <div className="legend-label">Normal</div>
                <div className="legend-bar" />
                <div className="legend-label">Oversize</div>
              </div>
              <div className="legend-notes">
                <span>
                  <span className="legend-dot ref" />
                  Reference disk
                </span>
                <span>
                  <span className="legend-dot alarm" />
                  Oversize chip
                </span>
              </div>
            </div>
          </div>
        {/* </div> */}

        <div className="right-stack">
          <div className="panel stats-panel" id="panel-stats">
            <h2>Live Statistics</h2>
            <StatsPanel data={stats} />
          </div>

          <div className="panel hist-panel" id="panel-hist">
            <div className="hist-header">
              <div className="hist-header-left">
                <h3 id="hist-title">Diameter Distribution</h3>
                <div className="hist-subtitle">Diameter-only distribution (rolling window)</div>
                <div className="hist-subtitle" id="hist-over-info">
                  {overInfo}
                </div>
              </div>
              <div className="hist-header-right">
                <span className="hist-subtitle">Type:</span>
                <select id="select-hist-type" disabled defaultValue="diameter">
                  <option value="diameter">Diameter</option>
                </select>
              </div>
            </div>
            <Histogram hist={hist?.diameter} ready={histReady} opts={histOpts} />
          </div>

          <div className="panel moist-panel" id="panel-moist">
            <h3>Moisture</h3>

            <div className="moist-grid">
              <div className="moist-kv-key">Status</div>
              <div className="moist-kv-val" id="moist-status">
                {mReady ? (
                  <span className="badge badge-ok">OK</span>
                ) : (
                  <span className="badge badge-warn">Collecting…</span>
                )}
              </div>

              <div className="moist-kv-key">Prediction</div>
              <div className="moist-kv-val" id="moist-pred">
                {mReady ? moist?.mean_pred || "—" : "—"}
              </div>

              <div className="moist-kv-key">Top-K</div>
              <div className="moist-kv-val" id="moist-topk">
                {mReady && moist?.topk != null ? String(moist.topk) : "—"}
              </div>

              <div className="moist-kv-key">Boxes used</div>
              <div className="moist-kv-val" id="moist-used">
                {mReady
                  ? String(
                      moist?.boxes_used != null
                        ? moist.boxes_used
                        : Array.isArray(moist?.per_box)
                        ? moist!.per_box!.length
                        : 0
                    )
                  : "—"}
              </div>

              <div className="moist-kv-key">Last update</div>
              <div className="moist-kv-val mono" id="moist-ts">
                {mReady && moist?.timestamp != null ? new Date(moist.timestamp * 1000).toLocaleString() : "—"}
              </div>
            </div>

            {mReady ? (
              <MoistureBars classes={classes} meanProbs={moist?.mean_probs} />
            ) : (
              <div className="prob-bars" id="moist-probs" />
            )}
            <div className="hist-subtitle" id="moist-classes">
              {mReady && classes.length > 0 ? "Classes: " + classes.join(", ") : "Classes: —"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
