# VTES OBS Plugin - Deployment Guide

## Quick Start

Run from **Developer Command Prompt for VS 2022** (or any terminal with MSBuild in PATH):

```cmd
deploy.bat
```

Or directly with PowerShell:

```powershell
.\deploy-windows.ps1
```

## Common Options

| Option | Description |
|--------|-------------|
| `-Configuration Release` | Build configuration (default: Release) |
| `-ObsInstallPath "C:\Program Files\obs-studio"` | OBS installation directory |
| `-ServerDeployPath "C:\VTES\vtes-server"` | Where to copy server files |
| `-SkipBuild` | Skip building, only deploy existing artifacts |
| `-SkipObsRestart` | Don't restart OBS after deployment |
| `-StartServer` | Auto-start vtes-server.exe after deployment |
| `-RunAsAdmin` | Force re-launch as Administrator |

## Examples

**Full deployment (build + deploy + restart OBS + start server):**
```powershell
.\deploy-windows.ps1 -StartServer
```

**Deploy only (skip build, use existing release artifacts):**
```powershell
.\deploy-windows.ps1 -SkipBuild
```

**Custom paths:**
```powershell
.\deploy-windows.ps1 -ObsInstallPath "D:\OBS" -ServerDeployPath "D:\VTES\server"
```

**CI/CD friendly (no restart, no server start):**
```powershell
.\deploy-windows.ps1 -SkipObsRestart
```

## What It Does

1. **Builds** the project via `build-windows.ps1` (CMake + MSBuild)
2. **Copies** plugin DLLs to `C:\Program Files\obs-studio\obs-plugins\64bit\`
3. **Copies** plugin data to `C:\Program Files\obs-studio\data\obs-plugins\vtes-card-scanner\`
4. **Copies** `vtes-server.exe`, `data\`, `public\` to deploy folder
5. **Restarts** OBS (if running)
6. **Starts** vtes-server.exe (with `-StartServer`)

## Requirements

- Windows 10/11
- Visual Studio 2022 with C++ workload
- CMake 3.20+
- OBS Studio installed
- Administrator privileges (for OBS Program Files access)

## Folder Structure Expected

```
project-root/
├── build-windows.ps1
├── deploy-windows.ps1
├── deploy.bat
├── CMakeLists.txt
├── src/                 # Plugin source
├── data/                # Plugin data (obs-plugins/vtes-card-scanner/)
├── public/              # Web assets for server
└── vtes-server/         # Server source
```

## Troubleshooting

**"Access denied" copying to OBS folder** → Run as Administrator

**"MSBuild not found"** → Open "Developer Command Prompt for VS 2022"

**"CMake not found"** → Install CMake and add to PATH

**OBS doesn't see plugin** → Verify DLL architecture matches OBS (64-bit)

**Server fails to start** → Check `data/` and `public/` copied correctly