import { useAuth } from "../context/AuthContext";
import { hasPerm } from "../lib/rbac";

interface Props {
  status: string;
  statusColor: string;
  onToggleControls: () => void;
}

export default function Header({ status, statusColor, onToggleControls }: Props) {
  const { user } = useAuth();
  const role = user?.role || "staff";
  return (
    <div className="header">
      <div className="header-left">
        <div className="header-title">Wood-chip Monitor</div>
        {/* <div className="header-subtitle">Live monitoring, events, quality rules, audit, and device health</div> */}
      </div>
      <div className="header-right">
        <div className="header-status" id="status-text" style={{ color: statusColor }}>
          {status}
        </div>
        {user && (
          <span className="pill" id="pill-user" style={{ display: "inline-block" }}>
            role: {role}
          </span>
        )}
        {user && hasPerm(role, "view_live") && (
          <button
            className="btn btn-ghost"
            id="btn-toggle-controls"
            style={{ display: "inline-flex" }}
            onClick={onToggleControls}
          >
            ⚙ Filters / Controls
          </button>
        )}
      </div>
    </div>
  );
}
