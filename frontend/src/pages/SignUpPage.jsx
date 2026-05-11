import { useState } from "react";
import { useNavigate } from "react-router-dom";
import SupabaseAuthCard from "../components/auth/SupabaseAuthCard";
import { supabase } from "../lib/supabaseClient";

export default function SignUpPage() {
  const navigate = useNavigate();
  const [errorMessage, setErrorMessage] = useState("");

  return (
    <SupabaseAuthCard
      mode="signup"
      supabase={supabase}
      errorMessage={errorMessage}
      onError={setErrorMessage}
      onNavigateSignIn={() => navigate("/login")}
    />
  );
}
