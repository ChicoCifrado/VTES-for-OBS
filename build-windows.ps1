# VTES Card Scanner - Windows Build Script
# Requires: Visual Studio Build Tools 2022/2026, CMake 3.16+
# Optional: vcpkg with opencv:x64-windows-static (recommended for full OpenCV modules)

param(
    [ValidateSet('Debug', 'RelWithDebInfo', 'Release', 'MinSizeRel')]
    [string] $Configuration = 'RelWithDebInfo',
    [switch] $BuildInstaller,
    [switch] $ForceFetchOpenCV
)

$ErrorActionPreference = 'Stop'

Write-Host "=== VTES Card Scanner Windows Build ===" -ForegroundColor Cyan
Write-Host "Configuration: $Configuration" -ForegroundColor Cyan

$ProjectRoot = $PSScriptRoot
$ServerRoot = "$ProjectRoot/../vtes-standalone-server"

# Detect vcpkg and system OpenCV
$UseSystemOpenCV = $false
$OpenCVConfigDir = $null
$OpenCVVersion = "unknown"
$VcpkgRoot = $env:VCPKG_ROOT
if (-not $VcpkgRoot) {
    # Common vcpkg locations
    $possiblePaths = @(
        "$env:USERPROFILE\vcpkg",
        "C:\vcpkg",
        "C:\tools\vcpkg",
        "$env:ProgramFiles\vcpkg"
    )
    foreach ($p in $possiblePaths) {
        # Check for both OpenCV 4 (calib3d.hpp) and OpenCV 5 (calib.hpp)
        # Prefer 5.x (calib.hpp) since both exist in OpenCV 5
        $calibPath = "$p\installed\x64-windows-static\include\opencv2\calib.hpp"
        $calib3dPath = "$p\installed\x64-windows-static\include\opencv2\calib3d.hpp"
        if ((Test-Path $calibPath) -or (Test-Path $calib3dPath)) {
            $VcpkgRoot = $p
            break
        }
    }
}
if ($VcpkgRoot -and -not $ForceFetchOpenCV) {
    $UseSystemOpenCV = $true
    Write-Host "vcpkg OpenCV detected at: $VcpkgRoot" -ForegroundColor Green
} elseif (Test-Path "C:\opencv\build\include\opencv2\calib.hpp") {
    # OpenCV 5.x detected (prefer 5.x over 4.x since both headers exist in 5)
    $UseSystemOpenCV = $true
    $OpenCVConfigDir = "C:/opencv/build"
    $OpenCVVersion = "5.x"
    Write-Host "System OpenCV 5.x detected at: C:\opencv" -ForegroundColor Green
    Write-Host "  OpenCVConfig.cmake at: $OpenCVConfigDir" -ForegroundColor Cyan
} elseif (Test-Path "C:\opencv\build\include\opencv2\calib3d.hpp") {
    # OpenCV 4.x detected
    $UseSystemOpenCV = $true
    $OpenCVConfigDir = "C:/opencv/build"
    $OpenCVVersion = "4.x"
    Write-Host "System OpenCV 4.x detected at: C:\opencv" -ForegroundColor Green
    Write-Host "  OpenCVConfig.cmake at: $OpenCVConfigDir" -ForegroundColor Cyan
} else {
    Write-Host "No system OpenCV found. Using FetchContent (limited modules)." -ForegroundColor Yellow
    Write-Host "  For full OpenCV, install vcpkg: `n  git clone https://github.com/microsoft/vcpkg && .\bootstrap-vcpkg.bat && .\vcpkg install opencv:x64-windows-static`n  Then set VCPKG_ROOT env var." -ForegroundColor Gray
}

# Step 1: Check for compiler
Write-Host "`n[0/6] Checking for C++ compiler..." -ForegroundColor Yellow
try {
    $clResult = cl 2>&1
    if ($LASTEXITCODE -ne 0 -and $clResult -match "not recognized") {
        Write-Host "ERROR: cl.exe not found in PATH." -ForegroundColor Red
        Write-Host "Run this from a 'Developer PowerShell for VS' or run:" -ForegroundColor Red
        Write-Host '& "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"' -ForegroundColor Red
        exit 1
    }
    Write-Host "Compiler found: OK" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Could not verify compiler. Build may fail." -ForegroundColor Yellow
}

# Step 2: Build OBS Plugin
Write-Host "`n[1/6] Configuring OBS Plugin..." -ForegroundColor Yellow
Push-Location $ProjectRoot
# Clean old build cache
if (Test-Path "build_x64") {
    Remove-Item -Recurse -Force "build_x64"
}

# Configure CMake with system OpenCV if available
$cmakeArgs = @("--preset", "windows-x64")
if ($UseSystemOpenCV) {
    $cmakeArgs += "-DUSE_SYSTEM_OPENCV=ON"
    Write-Host "Using system OpenCV (USE_SYSTEM_OPENCV=ON)" -ForegroundColor Cyan
    # Add vcpkg toolchain if vcpkg detected
    if ($VcpkgRoot -and (Test-Path "$VcpkgRoot\scripts\buildsystems\vcpkg.cmake")) {
        $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot\scripts\buildsystems\vcpkg.cmake"
        Write-Host "Using vcpkg toolchain: $VcpkgRoot\scripts\buildsystems\vcpkg.cmake" -ForegroundColor Cyan
    }
    # Add OpenCV_DIR for C:\opencv installation
    if ($OpenCVConfigDir) {
        $cmakeArgs += "-DOpenCV_DIR=$OpenCVConfigDir"
        Write-Host "Using OpenCV_DIR: $OpenCVConfigDir" -ForegroundColor Cyan
    }
}
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    throw "OBS Plugin CMake configuration failed!"
}

Write-Host "`n[2/6] Building OBS Plugin..." -ForegroundColor Yellow
cmake --build --preset windows-x64 --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    throw "OBS Plugin build failed!"
}

Write-Host "`n[3/6] Installing OBS Plugin..." -ForegroundColor Yellow
$ReleaseDir = "$ProjectRoot/release/$Configuration"
if (Test-Path $ReleaseDir) {
    Remove-Item -Recurse -Force $ReleaseDir
}
cmake --install "$ProjectRoot/build_x64" --prefix $ReleaseDir --config $Configuration
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    throw "OBS Plugin install failed!"
}

# Clean old obs-detect plugin from OBS installation
$ObsPluginDir = "C:\Program Files\obs-studio\obs-plugins\64bit"
$ObsDataDir = "C:\Program Files\obs-studio\data\obs-plugins"
if (Test-Path "$ObsPluginDir\obs-detect.dll") {
    Write-Host "Removing old obs-detect plugin..." -ForegroundColor Yellow
    Remove-Item "$ObsPluginDir\obs-detect.dll" -Force -ErrorAction SilentlyContinue
    Remove-Item "$ObsPluginDir\obs-detect.pdb" -Force -ErrorAction SilentlyContinue
}
if (Test-Path "$ObsDataDir\obs-detect") {
    Remove-Item "$ObsDataDir\obs-detect" -Recurse -Force -ErrorAction SilentlyContinue
}

# Install new plugin to OBS
Write-Host "Installing vtes-card-scanner to OBS..." -ForegroundColor Yellow
if (Test-Path "$ReleaseDir\obs-plugins\64bit\vtes-card-scanner.dll") {
    try {
        Copy-Item "$ReleaseDir\obs-plugins\64bit\vtes-card-scanner.dll" "$ObsPluginDir\" -Force -ErrorAction Stop
    } catch {
        Write-Host "WARNING: Could not copy DLL (OBS is probably running)." -ForegroundColor Yellow
        Write-Host "  Close OBS and copy manually from: $ReleaseDir\obs-plugins\64bit\" -ForegroundColor Yellow
    }
    if (Test-Path "$ReleaseDir\obs-plugins\64bit\vtes-card-scanner.pdb") {
        Copy-Item "$ReleaseDir\obs-plugins\64bit\vtes-card-scanner.pdb" "$ObsPluginDir\" -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path "$ReleaseDir\data\obs-plugins\vtes-card-scanner") {
        if (Test-Path "$ObsDataDir\vtes-card-scanner") {
            Remove-Item "$ObsDataDir\vtes-card-scanner" -Recurse -Force -ErrorAction SilentlyContinue
        }
        Copy-Item "$ReleaseDir\data\obs-plugins\vtes-card-scanner" "$ObsDataDir\" -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Copy OpenCV DLL (opencv_world) to plugin directory
    # Try OpenCV 5 first (opencv_world5xx.dll), then OpenCV 4 (opencv_world4xx.dll)
    $OpenCVDll = $null
    $cv5Candidates = Get-ChildItem "C:\opencv\build\x64\vc16\bin\opencv_world5*.dll" -ErrorAction SilentlyContinue
    if ($cv5Candidates) {
        $OpenCVDll = $cv5Candidates[0].FullName
    } else {
        $cv4Candidates = Get-ChildItem "C:\opencv\build\x64\vc16\bin\opencv_world4*.dll" -ErrorAction SilentlyContinue
        if ($cv4Candidates) {
            $OpenCVDll = $cv4Candidates[0].FullName
        }
    }
    if ($OpenCVDll -and (Test-Path $OpenCVDll)) {
        Copy-Item $OpenCVDll "$ObsPluginDir\" -Force -ErrorAction SilentlyContinue
        Write-Host "Copied OpenCV DLL: $(Split-Path $OpenCVDll -Leaf)" -ForegroundColor Green
    } else {
        Write-Host "WARNING: OpenCV DLL not found in C:\opencv\build\x64\vc16\bin\" -ForegroundColor Yellow
    }

    # Copy ONNX Runtime DLLs from build (lib/ and bin/)
    $OnnxDirs = @(
        "$ProjectRoot\build_x64\_deps\onnxruntime-src\lib",
        "$ProjectRoot\build_x64\_deps\onnxruntime-src\bin"
    )
    $copiedCount = 0
    foreach ($dir in $OnnxDirs) {
        if (Test-Path $dir) {
            Get-ChildItem "$dir\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
                Copy-Item $_.FullName "$ObsPluginDir\" -Force -ErrorAction SilentlyContinue
                Copy-Item $_.FullName "$ReleaseDir\obs-plugins\64bit\" -Force -ErrorAction SilentlyContinue
                Write-Host "  ONNX DLL: $($_.Name)" -ForegroundColor Green
                $copiedCount++
            }
        }
    }
    if ($copiedCount -eq 0) {
        Write-Host "WARNING: No ONNX Runtime DLLs found" -ForegroundColor Yellow
    } else {
        Write-Host "Copied $copiedCount ONNX Runtime DLLs" -ForegroundColor Green
    }

    Write-Host "OBS Plugin installed successfully!" -ForegroundColor Green
} else {
    Write-Host "WARNING: Plugin DLL not found in release folder." -ForegroundColor Yellow
}
Pop-Location

# Step 3: Build Standalone Server
Write-Host "`n[4/6] Building Standalone Server..." -ForegroundColor Yellow
if (Test-Path $ServerRoot) {
    Push-Location $ServerRoot
    if (Test-Path "build_x64") {
        Remove-Item -Recurse -Force "build_x64"
    }
    cmake -S . -B build_x64 -G "Visual Studio 18 2026" -A x64
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        throw "Server CMake configuration failed!"
    }
    cmake --build build_x64 --config $Configuration --parallel
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        throw "Server build failed!"
    }
    cmake --install build_x64 --prefix "$ReleaseDir" --config $Configuration
    Pop-Location
    Write-Host "Standalone Server: OK" -ForegroundColor Green
} else {
    Write-Host "Server source not found at $ServerRoot, skipping." -ForegroundColor Yellow
}

Write-Host "`n=== Build Complete ===" -ForegroundColor Green
Write-Host "Release files are in: $ReleaseDir" -ForegroundColor Green
Write-Host ""
Write-Host "To install manually:"
Write-Host "  1. Copy obs-plugins\64bit\*.dll to C:\Program Files\obs-studio\obs-plugins\64bit\"
Write-Host "  2. Copy data\obs-plugins\vtes-card-scanner\ to C:\Program Files\obs-studio\data\obs-plugins\vtes-card-scanner\"
Write-Host "  3. Copy vtes-server.exe, data\, and public\ to any folder"
Write-Host "  4. Run vtes-server.exe"
Write-Host "  5. Restart OBS"

# Step 4: Build installer (optional)
if ($BuildInstaller) {
    Write-Host "`n=== Building Installer ===" -ForegroundColor Cyan
    
    $IsccFile = "$ProjectRoot/build_x64/installer-Windows.generated.iss"
    if (-not (Test-Path $IsccFile)) {
        throw "InnoSetup script not found at: $IsccFile"
    }
    
    # Check if Inno Setup is installed
    $IsccPath = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $IsccPath)) {
        $IsccPath = "C:\Program Files\Inno Setup 6\ISCC.exe"
    }
    if (-not (Test-Path $IsccPath)) {
        Write-Host "`nInno Setup 6 not found. Download from: https://jrsoftware.org/isdl.php" -ForegroundColor Red
        Write-Host "Skipping installer build." -ForegroundColor Yellow
        return
    }
    
    Write-Host "Creating installer package..." -ForegroundColor Yellow
    
    $PackageDir = "$ProjectRoot/release/Package"
    if (Test-Path $PackageDir) {
        Remove-Item -Recurse -Force $PackageDir
    }
    Copy-Item -Path $ReleaseDir -Destination $PackageDir -Recurse
    
    Push-Location "$ProjectRoot/release"
    & $IsccPath $IsccFile /O"$ProjectRoot/release" /F"VTES-Card-Scanner-Installer"
    Pop-Location
    
    Remove-Item -Recurse -Force $PackageDir
    
    Write-Host "`n=== Installer Complete ===" -ForegroundColor Green
    Write-Host "Installer: $ProjectRoot/release/VTES-Card-Scanner-Installer.exe" -ForegroundColor Green
}
