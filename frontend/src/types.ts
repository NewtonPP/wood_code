// Shared API types (mirrors the backend payloads).

export type Role =
  | "staff"
  | "quality_engineer"
  | "software_engineer"
  | "manager"
  | "admin"
  | string;

export interface User {
  email: string;
  display_name?: string | null;
  role: Role;
}

export interface AdminUser {
  id: number;
  email: string;
  display_name?: string | null;
  role: Role;
  is_active: number;
  created_at: string;
}

export interface Stats {
  ready?: boolean;
  units?: string;
  mean_d?: number;
  median_d?: number;
  std_d?: number;
  min_d?: number;
  max_d?: number;
  // legacy fallbacks
  mean?: number;
  median?: number;
  std?: number;
  min?: number;
  max?: number;
  batch_label?: string;
  px_per_mm_mean?: number | null;
  px_per_mm_std?: number | null;
  alarm_active?: boolean;
  alarm_threshold_mm?: number | null;
  alarm_max_d_mm?: number | null;
  ref_diam_mm?: number;
}

export interface HistObj {
  bins: number[];
  counts: number[];
}

export interface HistData {
  ready?: boolean;
  units?: string;
  mode?: string;
  diameter?: HistObj | null;
  threshold_x?: number | null;
  pct_over_threshold?: number | null;
  alarm_threshold_mm?: number | null;
  alarm_enabled?: boolean;
}

export interface Moisture {
  ready?: boolean;
  timestamp?: number;
  classes?: string[] | null;
  mean_pred?: string;
  mean_pred_index?: number;
  mean_probs?: Record<string, number> | number[] | null;
  topk?: number;
  boxes_used?: number;
  per_box?: unknown[];
}

export interface RulesConfig {
  conf_thr?: number;
  nms_iou?: number;
  alarm_threshold_mm?: number;
  alarm_enabled?: boolean;
  ref_diam_mm?: number;
  histogram_mode?: string;
  moisture_enabled?: boolean;
  moisture_topk?: number;
  moisture_every_n_frames?: number;
  [k: string]: unknown;
}

export interface EventRow {
  id: number;
  ts: string;
  ts_epoch: number;
  device_id?: string | null;
  alarm_active: boolean;
  alarm_max_d_mm?: number | null;
  mean_d?: number | null;
  std_d?: number | null;
  units?: string | null;
  moisture_mean_pred?: string | null;
}

export interface RuleVersion {
  version: number;
  created_at: string;
  created_by_email?: string | null;
  reason?: string | null;
  applied: boolean;
  applied_at?: string | null;
}

export interface AuditItem {
  id: number;
  ts: string;
  user_email?: string | null;
  action: string;
  details: Record<string, unknown>;
}

// ---- Cloud inference (browser capture) ----
export interface InferBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  diameter: number;
  oversized: boolean;
}

export interface InferenceResult {
  ready?: boolean;
  units?: string;
  stats?: Stats | null;
  histogram?: HistData | null;
  moisture?: Moisture | null;
  boxes?: InferBox[];
  reference?: { cx: number; cy: number; diameter_px: number } | null;
  frame_size?: { w: number; h: number };
}

export interface IngestMessage {
  ok: boolean;
  error?: string;
  result?: InferenceResult;
}

export interface DeviceInfo {
  device_id: string;
  role: string; // "cloud" | "device"
}

export interface Health {
  ready?: boolean;
  camera_ok?: boolean;
  cam_dev?: string;
  last_frame_ts?: number | null;
  last_detr_ts?: number | null;
  fps_smoothed?: number | null;
  uptime_sec?: number | null;
  alarm_active?: boolean;
  last_error?: string | null;
}
