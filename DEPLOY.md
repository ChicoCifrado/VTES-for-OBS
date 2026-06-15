# VTES OBS Plugin - Deployment Guide

## Quick Start

Run from **Developer Command Prompt for VS 2022** (or any terminal with MSBuild in PATH):

```powershell
.\vtes-grimoire.ps1 build
.\vtes-grimoire.ps1 deploy
```

Or use the interactive menu:

```powershell
.\vtes-grimoire.ps1
```

## Build + Deploy in one step

```powershell
.\vtes-grimoire.ps1 all
```

## Package an installer (.exe)

```powershell
.\vtes-grimoire.ps1 build
.\vtes-grimoire.ps1 package
```

Requires [NSIS](https://nsis.sourceforge.io/Download) installed. Output in `release/`.

## What It Does

1. **Builds** the project via `cmake --preset windows-x64` (CMake + MSBuild)
2. **Copies** plugin DLLs to `C:\Program Files\obs-studio\obs-plugins\64bit\`
3. **Copies** plugin data to `C:\Program Files\obs-studio\data\obs-plugins\vtes-card-scanner\`
4. **Copies** ONNX Runtime + OpenCV DLLs alongside the plugin
5. **Restarts** OBS (if running)

## Requirements

- Windows 10/11
- Visual Studio 2022+ with C++ workload
- CMake 3.22+
- OBS Studio installed (64-bit)
- [NSIS](https://nsis.sourceforge.io/Download) (optional, for installer)
- Administrator privileges (for OBS Program Files access)

## Troubleshooting

**"Access denied" copying to OBS folder** → Run PowerShell as Administrator

**"MSBuild not found"** → Open "Developer Command Prompt for VS 2022"

**"CMake not found"** → Install CMake and add to PATH

**OBS doesn't see plugin** → Verify DLL architecture matches OBS (64-bit)