import { useEffect, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { Activity, Camera, Cpu, LogOut, ScrollText, ShieldCheck, SlidersHorizontal, Users } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { hasPerm } from "../lib/rbac";
import { getDeviceInfo } from "../lib/api";

interface Props {
  onLogout: () => void;
}

export default function Nav({ onLogout }: Props) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const role = user?.role || "staff";
  const [deviceId, setDeviceId] = useState("");

  useEffect(() => {
    getDeviceInfo().then((info) => setDeviceId(info.device_id));
  }, []);

  const handleLogout = async () => {
    await logout();
    onLogout();
    navigate("/login");
  };

  return (
    <div className="nav" id="nav" style={{ display: "flex" }}>
      <NavLink to="/live" id="nav-live">
        <Camera /> Live
      </NavLink>
      <NavLink to="/events" id="nav-events">
        <ScrollText /> Events
      </NavLink>
      {hasPerm(role, "edit_rules") && (
        <NavLink to="/quality" id="nav-quality">
          <SlidersHorizontal /> Quality Rules
        </NavLink>
      )}
      {hasPerm(role, "view_audit") && (
        <NavLink to="/audit" id="nav-audit">
          <ShieldCheck /> Audit
        </NavLink>
      )}
      {hasPerm(role, "view_devices") && (
        <NavLink to="/devices" id="nav-devices">
          <Activity /> Devices
        </NavLink>
      )}
      {hasPerm(role, "manage_users") && (
        <NavLink to="/admin/users" id="nav-admin-users">
          <Users /> Manage Users
        </NavLink>
      )}

      <div className="spacer" />
      <div className="nav-right">
        {deviceId && (
          <span className="hint" id="nav-device" title="This Jetson device" style={{ display: "inline-flex", alignItems: "center", gap: 4, marginRight: 10 }}>
            <Cpu size={14} /> {deviceId}
          </span>
        )}
        <button className="btn btn-ghost" id="btn-logout" onClick={handleLogout}>
          <LogOut /> Logout
        </button>
      </div>
    </div>
  );
}
