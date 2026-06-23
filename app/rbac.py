# app/rbac.py
"""
RBAC model (simple + extensible).
"""

ROLE_PERMS = {
    # Operator / Staff: can view live + events
    "staff": {"view_live", "view_events"},

    # QA: can export + edit rules
    "quality_engineer": {"view_live", "view_events", "export_events", "edit_rules", "view_audit"},

    # Software engineer: can see devices status + basic audit
    "software_engineer": {"view_live", "view_events", "export_events", "view_devices", "view_audit", "edit_rules"},

    # Manager: can do most things
    "manager": {"view_live", "view_events", "export_events", "edit_rules", "view_audit", "view_devices"},

    # Admin: everything (manage_users listed explicitly for clarity; "*" already covers it)
    "admin": {"*", "manage_users"},
}


def has_perm(role: str, perm: str) -> bool:
    perms = ROLE_PERMS.get(role, set())
    return ("*" in perms) or (perm in perms)
