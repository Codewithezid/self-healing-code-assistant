[CmdletBinding()]
param(
    [switch]$BuildRagIndex,
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Load-DotEnv {
    param(
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return
    }

    foreach ($line in Get-Content $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        if ($parts.Count -ne 2) {
            continue
        }
        $key = $parts[0].Trim()
        $value = $parts[1]
        Set-Item -Path "Env:$key" -Value $value
    }
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment..."
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv .venv
    } else {
        throw "Python is not installed or not on PATH."
    }
}

Write-Host "Installing dependencies..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

Load-DotEnv -Path ".env"

if (-not $env:MISTRAL_API_KEY) {
    Write-Host "Warning: MISTRAL_API_KEY is not set. The app will start, but model calls may fail." -ForegroundColor Yellow
}

if ($BuildRagIndex -and (Test-Path "scripts\index_project_rag.py")) {
    Write-Host "Building project RAG index..."
    & $venvPython "scripts\index_project_rag.py"
}

$port = 8000
if ($env:PORT) {
    $parsedPort = 0
    if ([int]::TryParse($env:PORT, [ref]$parsedPort)) {
        $port = $parsedPort
    }
}

$listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    if ($ForceRestart) {
        Write-Host "Port $port is in use by PID $($listener.OwningProcess). Stopping it..."
        Stop-Process -Id $listener.OwningProcess -Force
        Start-Sleep -Seconds 1
    } else {
        Write-Host "Port $port is already in use by PID $($listener.OwningProcess)." -ForegroundColor Yellow
        Write-Host "Use: .\\run_project.ps1 -ForceRestart" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "Starting app on http://127.0.0.1:$port ..."
& $venvPython "web_main.py"
