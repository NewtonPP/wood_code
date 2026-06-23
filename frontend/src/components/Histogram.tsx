import { useEffect, useRef } from "react";
import { useTheme } from "../context/ThemeContext";
import type { HistObj } from "../types";

interface Opts {
  thresholdX?: number | null;
  pctOver?: number | null;
  units?: string;
  thrLabelVal?: number | null;
}

interface Props {
  hist: HistObj | null | undefined;
  ready: boolean;
  opts?: Opts;
}

// Draw into logical (CSS-pixel) coordinates `w` x `h`. The caller scales the
// canvas backing store by devicePixelRatio so output stays crisp at any size.
function clearHistogram(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--bg-panel-2").trim() || "#101010";
  ctx.fillRect(0, 0, w, h);
}

function renderHistogram(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  histObj: HistObj,
  opts: Opts = {}
) {
  const bg = getComputedStyle(document.body).getPropertyValue("--bg-panel-2").trim() || "#101010";
  const border = getComputedStyle(document.body).getPropertyValue("--border").trim() || "#333";
  const label = getComputedStyle(document.body).getPropertyValue("--muted").trim() || "#888";

  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  if (!histObj || !histObj.bins || !histObj.counts || histObj.counts.length === 0) return;

  const bins = histObj.bins;
  const counts = histObj.counts;
  const n = counts.length;
  const maxCount = Math.max(...counts, 1);

  const marginLeft = 32,
    marginRight = 12,
    marginTop = 16,
    marginBottom = 28;
  const plotW = w - marginLeft - marginRight;
  const plotH = h - marginTop - marginBottom;
  if (plotW <= 0 || plotH <= 0) return;

  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(marginLeft, marginTop);
  ctx.lineTo(marginLeft, marginTop + plotH);
  ctx.lineTo(marginLeft + plotW, marginTop + plotH);
  ctx.stroke();

  const barW = plotW / n;
  const accent = getComputedStyle(document.body).getPropertyValue("--accent").trim() || "#4da6ff";
  for (let i = 0; i < n; i++) {
    const ratio = counts[i] / maxCount;
    const barH = ratio * (plotH - 2);
    const x = marginLeft + i * barW;
    const y = marginTop + plotH - barH;
    ctx.fillStyle = accent;
    ctx.fillRect(x + 1, y, Math.max(1, barW - 2), barH);
  }

  ctx.fillStyle = label;
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  // Fewer ticks on narrow charts so the x-axis labels don't collide.
  const numTicks = plotW < 320 ? 3 : 4;
  for (let t = 0; t <= numTicks; t++) {
    const idx = Math.round((t / numTicks) * (n - 1));
    const x = marginLeft + (idx + 0.5) * barW;
    ctx.fillText(Number(bins[idx]).toFixed(0), x, h - 9);
  }

  const thresholdX = opts.thresholdX != null ? Number(opts.thresholdX) : null;
  const pctOver = opts.pctOver != null ? Number(opts.pctOver) : null;
  const units = opts.units || "";

  if (thresholdX != null && Number.isFinite(thresholdX)) {
    const minX = Number(bins[0]);
    const maxX = Number(bins[n - 1]);
    const denom = maxX - minX;
    if (Number.isFinite(minX) && Number.isFinite(maxX) && Math.abs(denom) > 1e-9) {
      let r = (thresholdX - minX) / denom;
      r = Math.max(0, Math.min(1, r));
      const xLine = marginLeft + r * plotW;

      ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue("--error").trim() || "#ff6b6b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xLine, marginTop);
      ctx.lineTo(xLine, marginTop + plotH);
      ctx.stroke();

      const thrLabelVal = opts.thrLabelVal != null ? Number(opts.thrLabelVal) : thresholdX;
      const thrText = Number.isFinite(thrLabelVal)
        ? "Thr " + thrLabelVal.toFixed(0) + (units ? " " + units : "")
        : "Thr";
      const overText = pctOver != null && Number.isFinite(pctOver) ? "Over " + pctOver.toFixed(1) + "%" : "";
      const labelTxt = overText ? thrText + " | " + overText : thrText;

      ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--error").trim() || "#ff6b6b";
      ctx.font = "bold 13px sans-serif";
      const pad = 6;
      const placeRight = xLine + 120 < marginLeft + plotW;
      ctx.textAlign = placeRight ? "left" : "right";
      ctx.fillText(labelTxt, placeRight ? xLine + pad : xLine - pad, marginTop + 12);
    }
  }
}

export default function Histogram({ hist, ready, opts }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  const { theme } = useTheme();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Match the backing store to the element's real on-screen size so the
      // chart is never stretched, and multiply by DPR so it stays sharp on
      // high-density / mobile screens.
      const cssW = canvas.clientWidth;
      const cssH = canvas.clientHeight;
      if (cssW <= 0 || cssH <= 0) return;

      const dpr = window.devicePixelRatio || 1;
      const bw = Math.round(cssW * dpr);
      const bh = Math.round(cssH * dpr);
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      if (!ready || !hist || !hist.bins || !hist.counts) {
        clearHistogram(ctx, cssW, cssH);
        return;
      }
      renderHistogram(ctx, cssW, cssH, hist, opts || {});
    };

    draw();

    // Redraw whenever the canvas changes size (orientation change, panel
    // reflow, stacking at small breakpoints).
    const ro = new ResizeObserver(() => draw());
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [hist, ready, opts, theme]);

  return <canvas id="hist-diam" ref={ref} className="hist-canvas" />;
}
