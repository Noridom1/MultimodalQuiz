<#
.SYNOPSIS
Prints Render, Vercel, and Supabase dashboard values after you know both URLs.

.EXAMPLE
  .\scripts\Print-DeployEnvValues.ps1 -VercelOrigin "https://my-app.vercel.app" -RenderApiOrigin "https://multimodal-quiz-api.onrender.com"
#>
param(
    [Parameter(Mandatory = $true, HelpMessage = "Production SPA origin, e.g. https://my-app.vercel.app")]
    [string]$VercelOrigin,
    [Parameter(Mandatory = $true, HelpMessage = "Render API origin, e.g. https://multimodal-quiz-api.onrender.com")]
    [string]$RenderApiOrigin
)

function Normalize-Origin([string]$u) {
    $t = $u.Trim().TrimEnd('/')
    if ($t -notmatch '^https?://') {
        Write-Warning "Origin should include scheme (https://). Got: $u"
    }
    return $t
}

$v = Normalize-Origin $VercelOrigin
$r = Normalize-Origin $RenderApiOrigin

Write-Host ""
Write-Host "=== Render (Web Service > Environment) ===" -ForegroundColor Cyan
Write-Host "UI_ALLOWED_ORIGIN=$v"
Write-Host "(Optional, comma-separated) UI_ALLOWED_ORIGINS=<preview-url-1>,<preview-url-2>"
Write-Host "Set LLM and optional Supabase keys per repo .env.example (MISTRAL_API_KEY or OPENAI_API_KEY, QUIZGEN_*, SUPABASE_*, etc.)."
Write-Host ""
Write-Host "=== Vercel (Project > Settings > Environment Variables) ===" -ForegroundColor Cyan
Write-Host "VITE_API_BASE_URL=$r"
Write-Host "VITE_SUPABASE_URL=...   (from Supabase, if using auth)"
Write-Host "VITE_SUPABASE_ANON_KEY=..."
Write-Host "Redeploy the Vercel project after changing VITE_* variables."
Write-Host ""
Write-Host "=== Supabase (Authentication > URL configuration) ===" -ForegroundColor Cyan
Write-Host "Site URL: $v"
Write-Host "Redirect URLs: add $v and any paths you use after login (and preview URLs if needed)."
Write-Host ""
