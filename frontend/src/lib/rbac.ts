// Client-side RBAC mirror of the backend roles.
import type { Role } from "../types";

export const ROLE_PERMS: Record<string, Set<string>> = {
  staff: new Set(["view_live", "view_events"]),
  quality_engineer: new Set(["view_live", "view_events", "export_events", "edit_rules", "view_audit"]),
  software_engineer: new Set(["view_live", "view_events", "export_events", "view_devices", "view_audit", "edit_rules"]),
  manager: new Set(["view_live", "view_events", "export_events", "edit_rules", "view_audit", "view_devices"]),
  admin: new Set(["*", "manage_users"]),
};

export function hasPerm(role: Role, perm: string): boolean {
  const s = ROLE_PERMS[role] || new Set<string>();
  return s.has("*") || s.has(perm);
}
