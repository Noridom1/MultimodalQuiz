import { Navigate, Outlet, useLocation } from "react-router-dom";
import LoadingPanel from "../common/LoadingPanel";
import { useAuth } from "../../context/AuthContext";

export default function RequireAuth() {
  const { session, loading, supabaseConfigured } = useAuth();
  const location = useLocation();

  if (!supabaseConfigured) {
    return <Outlet />;
  }
  if (loading) {
    return (
      <div className="page-shell notebook-page">
        <LoadingPanel label="Checking session..." />
      </div>
    );
  }
  if (!session) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  return <Outlet />;
}
