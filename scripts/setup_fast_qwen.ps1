param(
    [string]$Ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Ollama)) {
    throw "No se encontró Ollama en '$Ollama'. Instálalo o pasa -Ollama <ruta>."
}

# Variante oficial Q4_K_M de unos 1,4 GB. Q2/Q3 externos empeoran tool calling.
& $Ollama pull qwen3:1.7b

$modelfile = Join-Path $PSScriptRoot "Modelfile.qwen3-1.7b-fast"
& $Ollama create qwen3:1.7b-fast -f $modelfile
if ($LASTEXITCODE -ne 0) { throw "No se pudo crear qwen3:1.7b-fast." }

Write-Host "Listo. En .env usa: OLLAMA_MODEL=qwen3:1.7b-fast"
