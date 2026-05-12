import { useId, useState } from "react";
import { motion } from "framer-motion";
import { Apple, Eye, EyeOff, Facebook, Loader2 } from "lucide-react";

function cn(...parts) {
  return parts.filter(Boolean).join(" ");
}

function GoogleGlyph({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="#4285F4"
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
      />
      <path
        fill="#34A853"
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
      />
      <path
        fill="#FBBC05"
        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
      />
      <path
        fill="#EA4335"
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
      />
    </svg>
  );
}

/**
 * @typedef {"signin" | "signup"} AuthMode
 */

/**
 * @param {object} props
 * @param {AuthMode} [props.mode]
 * @param {import("@supabase/supabase-js").SupabaseClient | null} [props.supabase]
 * @param {string} [props.redirectUrl]
 * @param {() => void} [props.onNavigateSignUp]
 * @param {() => void} [props.onNavigateSignIn]
 * @param {(msg: string) => void} [props.onError]
 * @param {string} [props.errorMessage]
 */
export default function SupabaseAuthCard({
  mode = "signin",
  supabase = null,
  redirectUrl = typeof window !== "undefined" ? `${window.location.origin}/` : undefined,
  onNavigateSignUp,
  onNavigateSignIn,
  onError,
  errorMessage = "",
}) {
  const isSignUp = mode === "signup";
  const emailId = useId();
  const passwordId = useId();
  const rememberId = useId();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [oauthBusy, setOauthBusy] = useState(/** @type {null | string} */ (null));
  const [formInfo, setFormInfo] = useState("");

  async function signInWithGoogle() {
    if (!supabase) {
      onError?.("Configure VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.");
      return;
    }
    setOauthBusy("google");
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: "google",
        options: { redirectTo: redirectUrl },
      });
      if (error) throw error;
    } catch (e) {
      onError?.(e?.message ?? "Google sign-in failed");
    } finally {
      setOauthBusy(null);
    }
  }

  async function signInWithApple() {
    if (!supabase) {
      onError?.("Configure VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.");
      return;
    }
    setOauthBusy("apple");
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: "apple",
        options: { redirectTo: redirectUrl },
      });
      if (error) throw error;
    } catch (e) {
      onError?.(e?.message ?? "Apple sign-in failed");
    } finally {
      setOauthBusy(null);
    }
  }

  async function signInWithFacebook() {
    if (!supabase) {
      onError?.("Configure VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.");
      return;
    }
    setOauthBusy("facebook");
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: "facebook",
        options: {
          redirectTo: redirectUrl,
          // Meta rejects `email` until it is enabled for the app (Permissions and Features).
          // `public_profile` works for new apps in Development without extra review.
          scopes: "public_profile",
        },
      });
      if (error) throw error;
    } catch (e) {
      onError?.(e?.message ?? "Facebook sign-in failed");
    } finally {
      setOauthBusy(null);
    }
  }

  async function handleForgotPassword(e) {
    e.preventDefault();
    if (!supabase) {
      onError?.("Supabase is not configured.");
      return;
    }
    if (!email.trim()) {
      onError?.("Enter your email, then tap Forgot password.");
      return;
    }
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email.trim(), {
        redirectTo: redirectUrl,
      });
      if (error) throw error;
      onError?.("");
      setFormInfo("If an account exists for this email, you will receive a reset link shortly.");
    } catch (err) {
      onError?.(err?.message ?? "Could not send reset email");
    }
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!supabase) {
      onError?.("Configure VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.");
      return;
    }
    setSubmitting(true);
    onError?.("");
    setFormInfo("");
    try {
      if (isSignUp) {
        const { error } = await supabase.auth.signUp({
          email: email.trim(),
          password,
          options: { emailRedirectTo: redirectUrl },
        });
        if (error) throw error;
        setFormInfo("Check your email to confirm your account if your project requires email verification.");
      } else {
        const { error } = await supabase.auth.signInWithPassword({
          email: email.trim(),
          password,
        });
        if (error) throw error;
        void rememberMe;
      }
    } catch (err) {
      onError?.(err?.message ?? (isSignUp ? "Sign up failed" : "Sign in failed"));
    } finally {
      setSubmitting(false);
    }
  }

  const title = isSignUp ? "Create your account" : "Welcome Back";
  const subtitle = isSignUp
    ? "Enter your details to get started with your workspace."
    : "Please enter your details to access your dashboard.";
  const primaryCta = isSignUp ? "Create Account" : "Sign In";

  return (
    <div className="tw-relative tw-flex tw-min-h-screen tw-w-full tw-overflow-hidden tw-bg-[#030712] tw-text-slate-100">
      <div
        className="tw-pointer-events-none tw-absolute tw-inset-0 tw-animate-gradient-shift tw-bg-gradient-to-br tw-from-[#020617] tw-via-[#0c1222] tw-to-[#042f2e]"
        style={{ backgroundSize: "200% 200%" }}
      />

      <motion.div
        aria-hidden
        className="tw-pointer-events-none tw-absolute tw--left-32 tw-top-1/4 tw-h-[420px] tw-w-[420px] tw-rounded-full tw-bg-cyan-500/25 tw-blur-[100px] tw-animate-pulse-slow"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.2 }}
      />
      <motion.div
        aria-hidden
        className="tw-pointer-events-none tw-absolute tw--right-40 tw-bottom-0 tw-h-[480px] tw-w-[480px] tw-rounded-full tw-bg-blue-600/20 tw-blur-[110px] tw-animate-pulse-slow"
        style={{ animationDelay: "2s" }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.4, delay: 0.15 }}
      />
      <motion.div
        aria-hidden
        className="tw-pointer-events-none tw-absolute tw-left-1/2 tw-top-0 tw-h-[300px] tw-w-[600px] tw--translate-x-1/2 tw-rounded-full tw-bg-cyan-400/10 tw-blur-[90px]"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay: 0.2 }}
      />

      <div className="tw-relative tw-z-10 tw-flex tw-w-full tw-flex-1 tw-items-center tw-justify-center tw-px-4 tw-py-12 sm:tw-px-6">
        <motion.div
          className="tw-w-full tw-max-w-[450px]"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
        >
          <div
            className={cn(
              "tw-relative tw-overflow-hidden tw-rounded-2xl tw-border tw-border-white/[0.08] tw-bg-slate-950/55 tw-p-8 tw-shadow-auth tw-backdrop-blur-2xl sm:tw-p-10",
              "tw-ring-1 tw-ring-cyan-400/10",
            )}
          >
            <div className="tw-pointer-events-none tw-absolute tw-inset-0 tw-bg-gradient-to-b tw-from-white/[0.04] tw-to-transparent" />

            <header className="tw-relative tw-mb-8 tw-text-center">
              <h1 className="tw-font-display tw-text-2xl tw-font-semibold tw-tracking-tight tw-text-white sm:tw-text-[1.65rem]">
                {title}
              </h1>
              <p className="tw-mt-2 tw-text-sm tw-leading-relaxed tw-text-slate-400">{subtitle}</p>
            </header>

            {errorMessage ? (
              <div
                className="tw-relative tw-mb-4 tw-rounded-lg tw-border tw-border-red-500/30 tw-bg-red-950/40 tw-px-3 tw-py-2 tw-text-sm tw-text-red-200"
                role="alert"
              >
                {errorMessage}
              </div>
            ) : null}

            {formInfo ? (
              <div
                className="tw-relative tw-mb-4 tw-rounded-lg tw-border tw-border-emerald-500/30 tw-bg-emerald-950/40 tw-px-3 tw-py-2 tw-text-sm tw-text-emerald-100"
                role="status"
              >
                {formInfo}
              </div>
            ) : null}

            <form className="tw-relative tw-space-y-5" onSubmit={handleSubmit} noValidate>
              <div className="tw-space-y-2">
                <label htmlFor={emailId} className="tw-text-sm tw-font-medium tw-text-slate-200">
                  Email
                </label>
                <input
                  id={emailId}
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className={cn(
                    "tw-flex tw-h-11 tw-w-full tw-rounded-lg tw-border tw-border-white/10 tw-bg-white/[0.04] tw-px-3 tw-py-2 tw-text-sm tw-text-white tw-outline-none tw-transition tw-duration-200",
                    "placeholder:tw-text-slate-500",
                    "focus-visible:tw-border-cyan-400/40 focus-visible:tw-ring-2 focus-visible:tw-ring-cyan-400/20",
                  )}
                  placeholder="you@company.com"
                />
              </div>

              <div className="tw-space-y-2">
                <div className="tw-flex tw-items-center tw-justify-between tw-gap-2">
                  <label htmlFor={passwordId} className="tw-text-sm tw-font-medium tw-text-slate-200">
                    Password
                  </label>
                  {!isSignUp ? (
                    <button
                      type="button"
                      onClick={handleForgotPassword}
                      className="tw-rounded-sm tw-text-xs tw-font-medium tw-text-cyan-400 tw-underline-offset-4 tw-transition hover:tw-text-cyan-300 hover:tw-underline focus-visible:tw-outline-none focus-visible:tw-ring-2 focus-visible:tw-ring-cyan-400/40"
                    >
                      Forgot Password?
                    </button>
                  ) : null}
                </div>
                <div className="tw-relative">
                  <input
                    id={passwordId}
                    name="password"
                    type={showPassword ? "text" : "password"}
                    autoComplete={isSignUp ? "new-password" : "current-password"}
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className={cn(
                      "tw-flex tw-h-11 tw-w-full tw-rounded-lg tw-border tw-border-white/10 tw-bg-white/[0.04] tw-py-2 tw-pl-3 tw-pr-11 tw-text-sm tw-text-white tw-outline-none tw-transition tw-duration-200",
                      "placeholder:tw-text-slate-500",
                      "focus-visible:tw-border-cyan-400/40 focus-visible:tw-ring-2 focus-visible:tw-ring-cyan-400/20",
                    )}
                    placeholder="••••••••"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((v) => !v)}
                    className="tw-absolute tw-right-1 tw-top-1/2 tw-flex tw-h-9 tw-w-9 tw--translate-y-1/2 tw-items-center tw-justify-center tw-rounded-md tw-text-slate-400 tw-transition hover:tw-bg-white/5 hover:tw-text-slate-200 focus-visible:tw-outline-none focus-visible:tw-ring-2 focus-visible:tw-ring-cyan-400/30"
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff className="tw-h-4 tw-w-4" /> : <Eye className="tw-h-4 tw-w-4" />}
                  </button>
                </div>
              </div>

              {!isSignUp ? (
                <div className="tw-flex tw-items-center tw-gap-2">
                  <input
                    id={rememberId}
                    name="remember"
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    className="tw-h-4 tw-w-4 tw-rounded tw-border-white/20 tw-bg-white/5 tw-text-cyan-500 tw-ring-offset-slate-950 focus-visible:tw-ring-2 focus-visible:tw-ring-cyan-400/40 focus-visible:tw-ring-offset-2"
                  />
                  <label htmlFor={rememberId} className="tw-select-none tw-text-sm tw-text-slate-400">
                    Remember me for 30 days
                  </label>
                </div>
              ) : null}

              <motion.button
                type="submit"
                disabled={submitting}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                className={cn(
                  "tw-relative tw-flex tw-h-12 tw-w-full tw-items-center tw-justify-center tw-overflow-hidden tw-rounded-lg tw-text-sm tw-font-semibold tw-text-slate-950 tw-transition",
                  "tw-bg-gradient-to-r tw-from-cyan-400 tw-via-cyan-500 tw-to-blue-600 tw-shadow-lg tw-shadow-cyan-500/20",
                  "hover:tw-shadow-xl hover:tw-shadow-cyan-500/25 focus-visible:tw-outline-none focus-visible:tw-ring-2 focus-visible:tw-ring-cyan-300/50 focus-visible:tw-ring-offset-2 focus-visible:tw-ring-offset-slate-950",
                  "disabled:tw-pointer-events-none disabled:tw-opacity-60",
                )}
              >
                <span className="tw-relative tw-z-10 tw-flex tw-items-center tw-gap-2">
                  {submitting ? (
                    <>
                      <Loader2 className="tw-h-4 tw-w-4 tw-animate-spin" aria-hidden />
                      {isSignUp ? "Creating…" : "Signing in…"}
                    </>
                  ) : (
                    primaryCta
                  )}
                </span>
                <span className="tw-pointer-events-none tw-absolute tw-inset-0 tw-bg-gradient-to-r tw-from-white/0 tw-via-white/25 tw-to-white/0 tw-opacity-0 tw-transition hover:tw-opacity-100" />
              </motion.button>
            </form>

            <div className="tw-relative tw-my-8">
              <div className="tw-absolute tw-inset-0 tw-flex tw-items-center" aria-hidden>
                <div className="tw-w-full tw-border-t tw-border-white/10" />
              </div>
              <div className="tw-relative tw-flex tw-justify-center tw-text-xs tw-uppercase tw-tracking-wider">
                <span className="tw-bg-slate-950/80 tw-px-3 tw-text-slate-500">Or continue with</span>
              </div>
            </div>

            <div className="tw-grid tw-grid-cols-1 tw-gap-3 sm:tw-grid-cols-3">
              <OAuthButton
                label="Google"
                busy={oauthBusy === "google"}
                onClick={signInWithGoogle}
                icon={<GoogleGlyph className="tw-h-[18px] tw-w-[18px]" />}
              />
              <OAuthButton
                label="Apple"
                busy={oauthBusy === "apple"}
                onClick={signInWithApple}
                icon={<Apple className="tw-h-[18px] tw-w-[18px]" strokeWidth={1.75} />}
              />
              <OAuthButton
                label="Facebook"
                busy={oauthBusy === "facebook"}
                onClick={signInWithFacebook}
                icon={<Facebook className="tw-h-[18px] tw-w-[18px]" fill="currentColor" strokeWidth={0} />}
              />
            </div>

            <p className="tw-relative tw-mt-8 tw-text-center tw-text-sm tw-text-slate-500">
              {isSignUp ? (
                <>
                  Already have an account?{" "}
                  <button
                    type="button"
                    onClick={onNavigateSignIn}
                    className="tw-rounded-sm tw-font-semibold tw-text-cyan-400 tw-underline-offset-4 tw-transition hover:tw-text-cyan-300 hover:tw-underline focus-visible:tw-outline-none focus-visible:tw-ring-2 focus-visible:tw-ring-cyan-400/40"
                  >
                    Sign in
                  </button>
                </>
              ) : (
                <>
                  Don&apos;t have an account?{" "}
                  <button
                    type="button"
                    onClick={onNavigateSignUp}
                    className="tw-rounded-sm tw-font-semibold tw-text-cyan-400 tw-underline-offset-4 tw-transition hover:tw-text-cyan-300 hover:tw-underline focus-visible:tw-outline-none focus-visible:tw-ring-2 focus-visible:tw-ring-cyan-400/40"
                  >
                    Create an Account
                  </button>
                </>
              )}
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

function OAuthButton({ label, icon, onClick, busy }) {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      disabled={busy}
      whileHover={{ y: -1 }}
      whileTap={{ scale: 0.98 }}
      className={cn(
        "tw-flex tw-h-11 tw-w-full tw-items-center tw-justify-center tw-gap-2 tw-rounded-lg tw-border tw-border-white/10",
        "tw-bg-white/[0.03] tw-text-sm tw-font-medium tw-text-slate-200 tw-backdrop-blur-sm tw-transition tw-duration-200",
        "hover:tw-border-cyan-400/25 hover:tw-bg-white/[0.06] hover:tw-shadow-oauth-hover",
        "focus-visible:tw-outline-none focus-visible:tw-ring-2 focus-visible:tw-ring-cyan-400/35",
        "disabled:tw-pointer-events-none disabled:tw-opacity-50",
      )}
    >
      {busy ? <Loader2 className="tw-h-4 tw-w-4 tw-animate-spin tw-text-cyan-400" /> : icon}
      {label}
    </motion.button>
  );
}
