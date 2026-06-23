// Thin API layer mirroring the original index.html fetch logic.
import type {
  AdminUser,
  AuditItem,
  EventRow,
  Health,
  HistData,
  Moisture,
  RuleVersion,
  RulesConfig,
  Stats,
  User,
} from "../types";

export async function apiFetch(url: string, opts: RequestInit = {}): Promise<Response> {
  const o: RequestInit = Object.assign({ cache: "no-store", credentials: "include" }, opts);
  return fetch(url, o);
}

const jsonHeaders = { "Content-Type": "application/json" };

// ---- Auth ----
export async function checkSession(): Promise<User | null> {
  try {
    const res = await apiFetch("/api/auth/me");
    if (!res.ok) return null;
    const data = await res.json();
    if (data && data.ok) return data.user as User;
    return null;
  } catch {
    return null;
  }
}

export async function loginRequest(email: string, password: string): Promise<Response> {
  return apiFetch("/api/auth/login", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ email, password }),
  });
}

export async function logoutRequest(): Promise<void> {
  try {
    await apiFetch("/api/auth/logout", { method: "POST" });
  } catch {
    /* ignore */
  }
}

export async function getDeviceInfo(): Promise<{ device_id: string }> {
  try {
    const res = await apiFetch("/api/info");
    if (!res.ok) return { device_id: "" };
    const data = await res.json();
    return { device_id: data?.device_id || "" };
  } catch {
    return { device_id: "" };
  }
}

export async function signupRequest(
  email: string,
  password: string,
  display_name: string
): Promise<Response> {
  return apiFetch("/api/auth/signup", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ email, password, display_name }),
  });
}

// ---- Admin: user management ----
export async function listUsers(): Promise<AdminUser[]> {
  const res = await apiFetch("/api/admin/users");
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return (data.users || []) as AdminUser[];
}

export async function createUser(
  email: string,
  password: string,
  role: string,
  display_name: string
): Promise<Response> {
  return apiFetch("/api/admin/users", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ email, password, role, display_name }),
  });
}

export async function updateUser(
  userId: number,
  patch: { role?: string; is_active?: boolean }
): Promise<Response> {
  return apiFetch("/api/admin/users/" + userId, {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(patch),
  });
}

// ---- Live ----
export const frameUrl = () => "/api/frame?ts=" + Date.now();

export async function getStats(): Promise<Stats> {
  const res = await apiFetch("/api/stats");
  if (!res.ok) throw new Error("stats " + res.status);
  return res.json();
}

export async function getHist(): Promise<HistData> {
  const res = await apiFetch("/api/hist");
  if (!res.ok) throw new Error("hist " + res.status);
  return res.json();
}

export async function getMoisture(): Promise<Moisture> {
  const res = await apiFetch("/api/moisture"); 
  if (!res.ok) throw new Error("moisture " + res.status);
  return res.json();
}

export async function getDeviceStatus(): Promise<Health> {
  const res = await apiFetch("/api/devices/status");
  if (!res.ok) throw new Error("devices " + (await res.text()));
  return res.json();
}

// ---- Rules / config ----
export async function getRulesCurrent(): Promise<{
  ok: boolean;
  source?: string;
  version?: number | null;
  rules: RulesConfig;
}> {
  const res = await apiFetch("/api/rules/current");
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateRules(rules: RulesConfig, reason: string): Promise<Response> {
  return apiFetch("/api/rules/update", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ rules, reason }),
  });
}

export async function getRuleVersions(limit = 80): Promise<{ versions: RuleVersion[] }> {
  const res = await apiFetch("/api/rules/versions?limit=" + limit);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function rollbackRules(version: number): Promise<Response> {
  return apiFetch("/api/rules/rollback", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ version }),
  });
}

// ---- Events ----
export async function getEvents(qs: URLSearchParams): Promise<{ events: EventRow[] }> {
  const res = await apiFetch("/api/events?" + qs.toString());
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export const eventsExportUrl = (qs: URLSearchParams) => "/api/events/export.csv?" + qs.toString();

// ---- Audit ----
export async function getAudit(limit: number): Promise<{ items: AuditItem[] }> {
  const res = await apiFetch("/api/audit?limit=" + encodeURIComponent(String(limit)));
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
