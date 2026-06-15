<# 
.SYNOPSIS
    Build and deploy VTES OBS plugin and server
.DESCRIPTION
    Automates building, deploying OBS plugin files, copying server files, and restarting OBS
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Configuration = 'Release',

    [Parameter(Mandatory=$false)]
    [string]$ProjectRoot = 'C:\Users\JackSuicide\VTES\vtes_obs_detect',

    [Parameter(Mandatory=$false)]
    [string]$ObsInstallPath = 'C:\Program Files\obs-studio',

    [Parameter(Mandatory=$false)]
    [string]$ServerDeployPath = 'C:\VTES\vtes-server',

    [Parameter(Mandatory=$false)]
    [switch]$SkipBuild,

    [Parameter(Mandatory=$false)]
    [switch]$SkipObsRestart,

    [Parameter(Mandatory=$false)]
    [switch]$StartServer,

    [Parameter(Mandatory=$false)]
    [switch]$RunAsAdmin
)

$ErrorActionPreference = 'Stop'

function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $timestamp = Get-Date -Format 'HH:mm:ss'
    $colors = @{ INFO='Green'; WARN='Yellow'; ERROR='Red'; DEBUG='Cyan' }
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $colors[$Level]
}

function Test-Admin {
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Ensure-Admin {
    if (-not (Test-Admin)) {
        Write-Log "This script requires Administrator privileges. Re-launching..." 'WARN'
        Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" @($args)" -Verb RunAs
        exit
    }
}

function Invoke-Build {
    if ($SkipBuild) {
        Write-Log "Skipping build (--SkipBuild specified)" 'WARN'
        return
    }

    Write-Log "Building project with configuration: $Configuration"
    $buildScript = Join-Path $ProjectRoot 'build-windows.ps1'
    
    if (-not (Test-Path $buildScript)) {
        Write-Log "build-windows.ps1 not found at $buildScript" 'ERROR'
        throw "Build script missing"
    }

    Push-Location $ProjectRoot
    & $buildScript -Configuration $Configuration
    $exitCode = $LASTEXITCODE
    Pop-Location
    
    if ($exitCode -ne 0) {
        throw "Build failed with exit code $exitCode"
    }
    Write-Log "Build completed successfully"
}

function Copy-ObsPlugins {
    Write-Log "Deploying OBS plugins..."

    $srcDllDir = Join-Path $ProjectRoot "release\$Configuration\obs-plugins\64bit"
    $dstDllDir = Join-Path $ObsInstallPath 'obs-plugins\64bit'
    
    if (-not (Test-Path $srcDllDir)) {
        Write-Log "Source DLL directory not found: $srcDllDir" 'ERROR'
        throw "Missing build output: obs-plugins\64bit"
    }

    if (-not (Test-Path $dstDllDir)) {
        Write-Log "Creating destination directory: $dstDllDir"
        New-Item -ItemType Directory -Path $dstDllDir -Force | Out-Null
    }

    $dlls = Get-ChildItem -Path $srcDllDir -Filter '*.dll'
    if ($dlls.Count -eq 0) {
        Write-Log "No DLLs found in $srcDllDir" 'WARN'
    } else {
        foreach ($dll in $dlls) {
            Write-Log "Copying $($dll.Name) to OBS plugins"
            Copy-Item $dll.FullName -Destination $dstDllDir -Force
        }
    }

    $srcDataDir = Join-Path $ProjectRoot "release\$Configuration\data\obs-plugins\vtes-card-scanner"
    $dstDataDir = Join-Path $ObsInstallPath 'data\obs-plugins\vtes-card-scanner'

    if (-not (Test-Path $srcDataDir)) {
        Write-Log "Source data directory not found: $srcDataDir" 'ERROR'
        throw "Missing build output: data\obs-plugins\vtes-card-scanner"
    }

    if (-not (Test-Path $dstDataDir)) {
        New-Item -ItemType Directory -Path $dstDataDir -Force | Out-Null
    }

    Write-Log "Copying vtes-card-scanner data files..."
    Copy-Item -Path (Join-Path $srcDataDir '*') -Destination $dstDataDir -Recurse -Force
    Write-Log "OBS plugin deployment completed"
}

function Copy-ServerFiles {
    Write-Log "Deploying server files to: $ServerDeployPath"

    if (-not (Test-Path $ServerDeployPath)) {
        New-Item -ItemType Directory -Path $ServerDeployPath -Force | Out-Null
    }

    $serverExe = Join-Path $ProjectRoot "release\$Configuration\vtes-server.exe"
    $srcData = Join-Path $ProjectRoot "release\$Configuration\data"
    $srcPublic = Join-Path $ProjectRoot "release\$Configuration\public"

    if (Test-Path $serverExe) {
        Write-Log "Copying vtes-server.exe"
        Copy-Item $serverExe -Destination $ServerDeployPath -Force
    } else {
        Write-Log "vtes-server.exe not found at $serverExe" 'WARN'
    }

    $dstData = Join-Path $ServerDeployPath 'data'
    if (Test-Path $srcData) {
        Write-Log "Copying data folder"
        if (-not (Test-Path $dstData)) {
            New-Item -ItemType Directory -Path $dstData -Force | Out-Null
        }
        Copy-Item -Path (Join-Path $srcData '*') -Destination $dstData -Recurse -Force
    } else {
        Write-Log "data folder not found at $srcData" 'WARN'
    }

    $dstPublic = Join-Path $ServerDeployPath 'public'
    if (Test-Path $srcPublic) {
        Write-Log "Copying public folder"
        if (-not (Test-Path $dstPublic)) {
            New-Item -ItemType Directory -Path $dstPublic -Force | Out-Null
        }
        Copy-Item -Path (Join-Path $srcPublic '*') -Destination $dstPublic -Recurse -Force
    } else {
        Write-Log "public folder not found at $srcPublic" 'WARN'
    }

    Write-Log "Server deployment completed"
}

function Restart-Obs {
    if ($SkipObsRestart) {
        Write-Log "Skipping OBS restart (--SkipObsRestart specified)" 'WARN'
        return
    }

    Write-Log "Restarting OBS..."
    $obsProcess = Get-Process -Name 'obs64' -ErrorAction SilentlyContinue
    if ($obsProcess) {
        Write-Log "Stopping existing OBS process..."
        Stop-Process -Name 'obs64' -Force
        Start-Sleep -Seconds 2
    }

    $obsExe = Join-Path $ObsInstallPath 'bin\64bit\obs64.exe'
    if (Test-Path $obsExe) {
        Write-Log "Starting OBS..."
        Start-Process -FilePath $obsExe -WindowStyle Normal
    } else {
        Write-Log "OBS executable not found at $obsExe" 'WARN'
    }
}

function Start-Server {
    if (-not $StartServer) {
        Write-Log "Skipping server start (use --StartServer to auto-start)" 'WARN'
        return
    }

    $serverExe = Join-Path $ServerDeployPath 'vtes-server.exe'
    if (Test-Path $serverExe) {
        Write-Log "Starting vtes-server.exe..."
        Start-Process -FilePath $serverExe -WorkingDirectory $ServerDeployPath -WindowStyle Normal
    } else {
        Write-Log "Server executable not found at $serverExe" 'ERROR'
    }
}

function Main {
    Write-Log "=== VTES OBS Deploy Script ==="
    Write-Log "Project Root: $ProjectRoot"
    Write-Log "Configuration: $Configuration"
    Write-Log "OBS Install Path: $ObsInstallPath"
    Write-Log "Server Deploy Path: $ServerDeployPath"

    if ($RunAsAdmin -or $PSVersionTable.Platform -eq 'Win32NT') {
        Ensure-Admin
    }

    try {
        Invoke-Build
        Copy-ObsPlugins
        Copy-ServerFiles
        Restart-Obs
        Start-Server
        Write-Log "=== Deployment completed successfully ===" -Level 'INFO'
    } catch {
        Write-Log "Deployment failed: $_" 'ERROR'
        exit 1
    }
}

Main