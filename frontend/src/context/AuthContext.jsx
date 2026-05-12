import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { configureApiClient } from "../api";
import { supabase } from "../lib/supabaseClient";

const AuthContext = createContext(null);

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}

export function AuthProvider({ children }) {
  const navigate = useNavigate();
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const tokenRef = useRef(null);

  const supabaseConfigured = Boolean(
    import.meta.env.VITE_SUPABASE_URL?.trim() && import.meta.env.VITE_SUPABASE_ANON_KEY?.trim(),
  );

  useEffect(() => {
    if (!supabase) {
      tokenRef.current = null;
      setSession(null);
      setLoading(false);
      return undefined;
    }
    let cancelled = false;
    supabase.auth.getSession().then(({ data: { session: next } }) => {
      if (cancelled) return;
      setSession(next ?? null);
      tokenRef.current = next?.access_token ?? null;
      setLoading(false);
    });
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next ?? null);
      tokenRef.current = next?.access_token ?? null;
    });
    return () => {
      cancelled = true;
      subscription.unsubscribe();
    };
  }, []);

  const signOut = useCallback(async () => {
    tokenRef.current = null;
    if (supabase) {
      await supabase.auth.signOut();
    }
    setSession(null);
    navigate("/login", { replace: true });
  }, [navigate]);

  useEffect(() => {
    configureApiClient({
      getToken: () => tokenRef.current,
      onAuthError: () => {
        tokenRef.current = null;
        setSession(null);
        if (supabase) {
          void supabase.auth.signOut();
        }
        navigate("/login", { replace: true });
      },
    });
  }, [navigate]);

  const value = useMemo(
    () => ({
      session,
      user: session?.user ?? null,
      loading,
      signOut,
      supabaseConfigured,
    }),
    [session, loading, signOut, supabaseConfigured],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
