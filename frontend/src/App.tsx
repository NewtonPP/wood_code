import { useCallback, useState, type ReactElement } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Nav from "./components/Nav";
import ControlsDrawer from "./components/ControlsDrawer";
import Tour from "./components/Tour";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import LivePage from "./pages/LivePage";
import EventsPage from "./pages/EventsPage";
import QualityPage from "./pages/QualityPage";
import AuditPage from "./pages/AuditPage";
import DevicesPage from "./pages/DevicesPage";
import AdminUsersPage from "./pages/AdminUsersPage";
import { useAuth } from "./context/AuthContext";
import { hasPerm } from "./lib/rbac";
import { usePolling } from "./hooks/usePolling";
import { apiFetch } from "./lib/api";
import type { Stats } from "./types";

export default function App() {
  const { user, ready } = useAuth();
  const [controlsOpen, setControlsOpen] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);

  const openControls = useCallback(() => setControlsOpen(true), []);
  const closeControls = useCallback(() => setControlsOpen(false), []);
  const isControlsOpen = useCallback(
    () => document.getElementById("controls-backdrop")?.classList.contains("open") ?? false,
    []
  );

  // Poll stats for the header status + Live panel (1 Hz), like the original.
  usePolling(
    async () => {
      if (!user) return;
      try {
        const res = await apiFetch("/api/stats");
        if (!res.ok) return;
        const data = await res.json();
        setStats(data);
      } catch {
        /* ignore — Live page shows "Collecting…" until stats arrive */
      }
    },
    1000,
    !!user
  );

  const role = user?.role || "staff";
  const defaultRoute = user ? (hasPerm(role, "view_live") ? "/live" : "/events") : "/login";

  const guard = (perm: string, element: ReactElement): ReactElement => {
    if (!user) return <Navigate to="/login" replace />;
    if (!hasPerm(role, perm)) return <Navigate to={defaultRoute} replace />;
    return element;
  };

  return (
    <>
      <div id="app-shell">
        {user && <Nav onLogout={closeControls} />}

        <div className="main">
          {ready && (
            <Routes>
              <Route path="/login" element={user ? <Navigate to={defaultRoute} replace /> : <LoginPage />} />
              <Route path="/signup" element={user ? <Navigate to={defaultRoute} replace /> : <SignupPage />} />
              <Route path="/live" element={guard("view_live", <LivePage stats={stats} />)} />
              <Route path="/events" element={guard("view_events", <EventsPage />)} />
              <Route path="/quality" element={guard("edit_rules", <QualityPage />)} />
              <Route path="/audit" element={guard("view_audit", <AuditPage />)} />
              <Route path="/devices" element={guard("view_devices", <DevicesPage />)} />
              <Route path="/admin/users" element={guard("manage_users", <AdminUsersPage />)} />
              <Route path="*" element={<Navigate to={defaultRoute} replace />} />
            </Routes>
          )}
        </div>

        <ControlsDrawer open={controlsOpen} onClose={closeControls} />
      </div>

      <Tour
        armed={!!user && ready}
        openControls={openControls}
        closeControls={closeControls}
        isControlsOpen={isControlsOpen}
      />
    </>
  );
}
