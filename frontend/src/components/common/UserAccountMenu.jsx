import { useEffect, useId, useRef, useState } from "react";
import { ChevronDown, LogOut, User } from "lucide-react";
import { useAuth } from "../../context/AuthContext";

function initialsFromUser(user) {
  const email = user?.email || "";
  const meta = user?.user_metadata || {};
  const name = typeof meta.full_name === "string" ? meta.full_name : "";
  if (name.trim()) {
    const parts = name.trim().split(/\s+/);
    const a = parts[0]?.[0] || "";
    const b = parts[1]?.[0] || "";
    return (a + b).toUpperCase() || a.toUpperCase() || "?";
  }
  if (email) {
    return email.slice(0, 2).toUpperCase();
  }
  return "?";
}

export default function UserAccountMenu() {
  const { user, signOut } = useAuth();
  const [open, setOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const wrapRef = useRef(null);
  const menuId = useId();

  useEffect(() => {
    if (!open) return undefined;
    function onPointerDown(event) {
      if (!wrapRef.current?.contains(event.target)) {
        setOpen(false);
      }
    }
    function onKey(event) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!user) {
    return null;
  }

  const initials = initialsFromUser(user);

  return (
    <div className="user-account-menu" ref={wrapRef}>
      <button
        type="button"
        className="user-account-trigger"
        aria-expanded={open}
        aria-haspopup="true"
        aria-controls={menuId}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="user-account-avatar" aria-hidden>
          {initials}
        </span>
        <ChevronDown size={16} className={open ? "user-account-chevron open" : "user-account-chevron"} />
      </button>

      {open ? (
        <div id={menuId} className="user-account-dropdown" role="menu">
          <button
            type="button"
            className="user-account-item"
            role="menuitem"
            onClick={() => {
              setOpen(false);
              setProfileOpen(true);
            }}
          >
            <User size={18} />
            View profile
          </button>
          <button
            type="button"
            className="user-account-item danger"
            role="menuitem"
            onClick={() => {
              setOpen(false);
              void signOut();
            }}
          >
            <LogOut size={18} />
            Sign out
          </button>
        </div>
      ) : null}

      {profileOpen ? (
        <div
          className="user-profile-modal-overlay"
          role="presentation"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setProfileOpen(false);
          }}
        >
          <div className="user-profile-modal" role="dialog" aria-labelledby="user-profile-title">
            <h2 id="user-profile-title">Your profile</h2>
            <dl className="user-profile-fields">
              <dt>Email</dt>
              <dd>{user.email || "—"}</dd>
              <dt>User ID</dt>
              <dd className="user-profile-mono">{user.id}</dd>
            </dl>
            <button type="button" className="primary-pill compact" onClick={() => setProfileOpen(false)}>
              Close
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
