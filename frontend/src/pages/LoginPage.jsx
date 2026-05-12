import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import SupabaseAuthCard from "../components/auth/SupabaseAuthCard";
import { supabase } from "../lib/supabaseClient";

function safeReturnPath(state) {
  const raw = typeof state?.from === "string" ? state.from : "/";
  if (!raw.startsWith("/") || raw.startsWith("//")) {
    return "/";
  }
  return raw;
}

export default function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [errorMessage, setErrorMessage] = useState("");
  const returnTo = safeReturnPath(location.state);

  useEffect(() => {
    if (!supabase) return undefined;
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === "SIGNED_IN" && session) {
        navigate(returnTo, { replace: true });
      }
    });
    return () => subscription.unsubscribe();
  }, [navigate, returnTo]);

  return (
    <SupabaseAuthCard
      mode="signin"
      supabase={supabase}
      errorMessage={errorMessage}
      onError={setErrorMessage}
      onNavigateSignUp={() => navigate("/signup")}
    />
  );
}
