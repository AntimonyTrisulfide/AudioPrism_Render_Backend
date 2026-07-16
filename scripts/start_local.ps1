param(
    [int]$Port = 8001,
    [switch]$Reload,
    [switch]$Restart,
    [switch]$LocalJsonAuth
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Resolve-Path (Join-Path $ScriptDir "..")
$LocalPython = Join-Path $BackendRoot ".venv\Scripts\python.exe"
$WorkspacePython = Resolve-Path -LiteralPath (Join-Path $BackendRoot "..\..\.venv\Scripts\python.exe") -ErrorAction SilentlyContinue

if (Test-Path -LiteralPath $LocalPython) {
    $Python = $LocalPython
} elseif ($WorkspacePython) {
    $Python = $WorkspacePython.Path
} else {
    Write-Host "No Python 3.11 venv found. Creating .venv in backend repo..."
    Push-Location $BackendRoot
    try {
        py -3.11 -m venv .venv
        .\.venv\Scripts\python.exe -m pip install -r requirements.txt
        $Python = Join-Path $BackendRoot ".venv\Scripts\python.exe"
    } finally {
        Pop-Location
    }
}

$Version = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($Version -ne "3.11") {
    throw "AudioPrism local backend must run on Python 3.11. Found Python $Version at $Python"
}

if ($LocalJsonAuth) {
    $env:MONGO_REQUIRED = "0"
    $env:MONGODB_URI = ""
    $env:MONGO_URI = ""
    $env:AUTH_REQUIRED = "1"
    $env:STORAGE_BACKEND = "local"
    $env:RESULT_PERSIST_REQUIRED = "0"
    $env:SUPABASE_URL = ""
    $env:SUPABASE_SERVICE_ROLE_KEY = ""
    Write-Host "Local test mode enabled: auth/history use local JSON and stems use local /output."
}

function Get-ListeningProcessId {
    param([int]$LocalPort)

    try {
        $Lines = netstat -ano
        foreach ($Line in $Lines) {
            if ($Line -match "^\s*TCP\s+127\.0\.0\.1:$LocalPort\s+\S+\s+LISTENING\s+(\d+)\s*$") {
                return [int]$Matches[1]
            }
            if ($Line -match "^\s*TCP\s+0\.0\.0\.0:$LocalPort\s+\S+\s+LISTENING\s+(\d+)\s*$") {
                return [int]$Matches[1]
            }
        }
    } catch {
        # Fall through to Get-NetTCPConnection.
    }

    try {
        $Connection = Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction Stop |
            Select-Object -First 1
        if ($Connection) {
            return [int]$Connection.OwningProcess
        }
    } catch {
        return $null
    }
    return $null
}

$ExistingPid = Get-ListeningProcessId -LocalPort $Port
if ($ExistingPid) {
    if (-not $Restart) {
        try {
            $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/healthz" -TimeoutSec 2
            if ($Health.status -eq "ok") {
                Write-Host "AudioPrism backend is already running on http://127.0.0.1:$Port (PID $ExistingPid)."
                Write-Host "Use .\start_backend.bat -Restart if you want to restart it."
                if ($LocalJsonAuth) {
                    Write-Host "If you need local test mode, run .\start_backend.bat -Restart -LocalJsonAuth."
                }
                exit 0
            }
        } catch {
            Write-Host "Port $Port is already in use by PID $ExistingPid, but /healthz did not respond."
            Write-Host "Use .\start_backend.bat -Restart to stop that process and start AudioPrism."
            exit 1
        }

        Write-Host "Port $Port is already in use by PID $ExistingPid."
        Write-Host "Use .\start_backend.bat -Restart if you want to restart it."
        exit 1
    }

    Write-Host "Stopping existing process on port $Port (PID $ExistingPid)..."
    Stop-Process -Id $ExistingPid -Force
    Start-Sleep -Seconds 1
}

$Args = @("-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "$Port")
if ($Reload) {
    $Args += "--reload"
}

Push-Location $BackendRoot
try {
    & $Python @Args
} finally {
    Pop-Location
}
