import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import SupabaseAuthCard from "../components/auth/SupabaseAuthCard";
import { supabase } from "../lib/supabaseClient";

export default function LoginPage() {
  const navigate = useNavigate();
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    if (!supabase) return undefined;
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === "SIGNED_IN" && session) {
        navigate("/", { replace: true });
      }
    });
    return () => subscription.unsubscribe();
  }, [navigate]);

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
