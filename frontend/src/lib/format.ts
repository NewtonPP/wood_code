// Formatting helpers ported from the original index.html.

export function parseLocalToEpoch(s: string): number | null {
  // Accept "YYYY-MM-DD HH:MM" in local time
  const t = (s || "").trim();
  if (!t) return null;
  const m = t.match(/^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})$/);
  if (!m) return null;
  const [, yy, mm, dd, hh, mi] = m;
  const dt = new Date(Number(yy), Number(mm) - 1, Number(dd), Number(hh), Number(mi), 0);
  return Math.floor(dt.getTime() / 1000);
}

export function fmtNum(x: unknown, digits = 1): string {
  if (x == null || x === "") return "—";
  const v = Number(x);
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

export function fmtTs(tsEpoch: unknown): string {
  if (tsEpoch == null) return "—";
  const v = Number(tsEpoch);
  if (!Number.isFinite(v)) return "—";
  return new Date(v * 1000).toLocaleString();
}

export function fmtUptime(sec: unknown): string {
  if (sec == null) return "—";
  const s = Math.max(0, Math.floor(Number(sec)));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m ${r}s`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}
