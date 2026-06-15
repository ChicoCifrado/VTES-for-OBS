#!/usr/bin/env pwsh
<#
.SYNOPSIS
    VTES Card Scanner - Grimorio Sanguineo
    Unified TUI for building, deploying, and managing the VTES OBS plugin.
#>

#requires -Version 5.1

$ErrorActionPreference = "Stop"

# Colors
$C = @{
    CRIMSON     = [char]27 + "[38;2;139;0;0m"
    DARK_CRIMSON = [char]27 + "[38;2;100;0;0m"
    GOLD        = [char]27 + "[38;2;184;134;11m"
    LIGHT_GOLD  = [char]27 + "[38;2;218;165;32m"
    SILVER      = [char]27 + "[38;2;192;192;192m"
    WHITE       = [char]27 + "[38;2;230;230;230m"
    RED_BG      = [char]27 + "[48;2;139;0;0m"
    DIM         = [char]27 + "[2m"
    BOLD        = [char]27 + "[1m"
    RESET       = [char]27 + "[0m"
    CLEAR       = [char]27 + "[2J" + [char]27 + "[H"
    INVERT      = [char]27 + "[7m"
    UNDERLINE   = [char]27 + "[4m"
}

$SCRIPT:PROJECT_ROOT    = Split-Path -Parent $PSCommandPath
$SCRIPT:OBS_PLUGIN_DIR  = "C:\Program Files\obs-studio\obs-plugins\64bit"
$SCRIPT:OBS_DATA_DIR    = "C:\Program Files\obs-studio\data\obs-plugins\vtes-card-scanner"
$SCRIPT:BUILD_DIR       = Join-Path $SCRIPT:PROJECT_ROOT "build_x64"
$SCRIPT:RELEASE_DIR     = Join-Path $SCRIPT:PROJECT_ROOT "release"
$SCRIPT:PER_TYPE_DIR    = Join-Path $SCRIPT:PROJECT_ROOT "data\per_type"
$SCRIPT:MANIFEST_PATH   = Join-Path $SCRIPT:PER_TYPE_DIR "per_type_manifest.json"
$SCRIPT:TESSERACT_PATH  = "C:\Program Files\Tesseract-OCR"
$SCRIPT:WSL_DATA_DIR    = "/mnt/d/Hermes/vtes/data/per_type"
$SCRIPT:WSL_MANIFEST    = "/mnt/d/Hermes/vtes/data/per_type_manifest.json"

function Show-Ankh {
    $artPath = Join-Path $SCRIPT:PROJECT_ROOT "ankh.txt"
    if (Test-Path $artPath) {
        $art = Get-Content $artPath -Raw
        Write-Host ($C.BOLD + $C.CRIMSON + $art + $C.RESET)
    }
}

function Write-Grim {
    param([string]$Text, [string]$Color = "SILVER")
    Write-Host ($C[$Color] + $Text + $C.RESET)
}

function Write-BoldGrim {
    param([string]$Text, [string]$Color = "CRIMSON")
    Write-Host ($C.BOLD + $C[$Color] + $Text + $C.RESET)
}

function Write-DimGrim {
    param([string]$Text)
    Write-Host ($C.DIM + $C.SILVER + $Text + $C.RESET)
}

function Write-Banner {
    param([string]$Text)
    $pad = "=" * 40
    Write-Host ($C.CRIMSON + "+ " + $C.GOLD + $C.BOLD + $Text + $C.RESET + " " + $C.DIM + $C.CRIMSON + $pad + $C.RESET + " +" + $C.RESET)
}

function Write-Success { Write-Host ($C.BOLD + $C.GOLD + ">" + $C.RESET + " " + $C.SILVER + $args[0] + $C.RESET) }
function Write-Error  { Write-Host ($C.BOLD + $C.CRIMSON + "x" + $C.RESET + " " + $C.SILVER + $args[0] + $C.RESET) }
function Write-Info   { Write-Host ($C.DIM + $C.SILVER + "o" + $C.RESET + " " + $C.DIM + $args[0] + $C.RESET) }

function Show-Separator {
    Write-Host ("  " + $C.DARK_CRIMSON + ("=" * 50) + $C.RESET)
}

function Write-FrameTop {
    Write-Host ($C.CRIMSON + "+" + ("=" * 55) + "+" + $C.RESET)
}

function Write-FrameMid {
    Write-Host ($C.CRIMSON + "|" + $C.RESET + $args[0] + $C.CRIMSON + "|" + $C.RESET)
}

function Write-FrameSep {
    Write-Host ($C.CRIMSON + "+" + ("=" * 55) + "+" + $C.RESET)
}

function Write-FrameBot {
    Write-Host ($C.CRIMSON + "+" + ("=" * 55) + "+" + $C.RESET)
}

function Show-Header {
    $width = 55
    $title  = "V T E S   C A R D   S C A N N E R"
    $sub    = "-- Grimorio Sanguineo --"
    $ver    = "v1.0.0"

    $tPad   = [math]::Floor(($width - $title.Length) / 2)
    $sPad   = [math]::Floor(($width - $sub.Length) / 2)

    Write-FrameTop
    Write-FrameMid ("  " + (" " * $tPad) + $C.BOLD + $C.CRIMSON + $title + $C.RESET + "  ")
    Write-FrameMid ("  " + (" " * $sPad) + $C.DIM + $C.GOLD + $sub + $C.RESET + "  ")
    Write-FrameBot
}

function Update-StatusLine {
    param([string]$BuildStatus = "?", [string]$ModelsStatus = "?", [string]$TesseractStatus = "?")
    function StatusDot {
        param([string]$S)
        if ($S -eq "OK" -or $S -eq "true" -or $S -eq $true) { return ($C.BOLD + $C.GOLD + "O" + $C.RESET) }
        else { return ($C.DIM + $C.SILVER + "o" + $C.RESET) }
    }
    $bd = StatusDot $BuildStatus
    $md = StatusDot $ModelsStatus
    $td = StatusDot $TesseractStatus
    Write-Host (" " + $C.CRIMSON + "|" + $C.RESET + "  Build " + $bd + "  Models " + $md + "  OCR " + $td + "  " + $C.CRIMSON + "|" + $C.RESET)
}

function Test-BuildReady {
    $dll = Join-Path $SCRIPT:OBS_PLUGIN_DIR "vtes-card-scanner.dll"
    return (Test-Path $dll)
}

function Test-ModelsReady {
    if (-not (Test-Path $SCRIPT:MANIFEST_PATH)) { return $false }
    try {
        $m = Get-Content $SCRIPT:MANIFEST_PATH -Raw | ConvertFrom-Json
        $count = ($m.PSObject.Properties | Measure-Object).Count
        return $count -eq 14
    } catch { return $false }
}

function Test-TesseractReady {
    $exe = Join-Path $SCRIPT:TESSERACT_PATH "tesseract.exe"
    $tdata = Join-Path $SCRIPT:TESSERACT_PATH "tessdata\eng.traineddata"
    return (Test-Path $exe) -and (Test-Path $tdata)
}

function Test-OcrCompiled {
    $dll = Join-Path $SCRIPT:OBS_PLUGIN_DIR "vtes-card-scanner.dll"
    if (-not (Test-Path $dll)) { return $false }
    $bytes = [System.IO.File]::ReadAllBytes($dll)
    $text = [System.Text.Encoding]::ASCII.GetString($bytes)
    return $text -match "Tesseract|VTES_HAVE_TESSERACT|tesseract"
}

function Get-CalculationSummary {
    $totalClasses = 0
    if (Test-Path $SCRIPT:MANIFEST_PATH) {
        try {
            $m = Get-Content $SCRIPT:MANIFEST_PATH -Raw | ConvertFrom-Json
            foreach ($prop in $m.PSObject.Properties) {
                $totalClasses += $prop.Value.classes
            }
        } catch {}
    }
    return $totalClasses
}

function Invoke-Verify {
    Clear-Host
    Write-FrameTop
    Write-FrameMid ("  " + $C.BOLD + $C.GOLD + "RITUAL DE VERIFICACION" + $C.RESET)
    Write-FrameSep
    Write-Host ""

    $built = Test-BuildReady
    if ($built) {
        Write-Success "Plugin instalado en OBS"
        $dllVer = (Get-Item (Join-Path $SCRIPT:OBS_PLUGIN_DIR "vtes-card-scanner.dll")).Length
        Write-Info ("  DLL: " + [math]::Round($dllVer / 1MB, 1) + " MB")
    } else {
        Write-Error "Plugin NO instalado en OBS"
    }

    $modelsOK = Test-ModelsReady
    if ($modelsOK) {
        $count = (Get-ChildItem $SCRIPT:PER_TYPE_DIR -Filter "*.onnx").Count
        $total = Get-CalculationSummary
        $size  = [math]::Round((Get-ChildItem $SCRIPT:PER_TYPE_DIR -Recurse | Where-Object { -not $_.PSIsContainer } | Measure-Object -Property Length -Sum).Sum / 1MB, 0)
        Write-Success ("Modelos per-type: " + $count + " ONNX, " + $total + " clases, " + $size + " MB")
    } else {
        Write-Error "Modelos per-type NO encontrados (esperados: 14)"
    }

    $tess = Test-TesseractReady
    if ($tess) {
        Write-Success ("Tesseract OCR instalado (" + $SCRIPT:TESSERACT_PATH + ")")
    } else {
        Write-Error "Tesseract OCR NO instalado"
    }

    $ocrBuilt = $false
    if ($built) {
        $ocrBuilt = Test-OcrCompiled
        if ($ocrBuilt) { Write-Success "Plugin compilado CON soporte OCR" }
        else { Write-Info "Plugin compilado SIN soporte OCR (usa -DUSE_SYSTEM_TESSERACT=ON)" }
    }

    Write-Host ""
    Write-FrameSep
    Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + "RESUMEN" + $C.RESET)
    Write-FrameMid ("  " + $C.DIM + $C.SILVER + "Pipeline: YOLO -> Type Classifier -> Per-Type Embedder -> Global Embedder" + $C.RESET)
    Write-FrameMid ("  " + $C.DIM + $C.SILVER + "          -> OCR Fallback (cuando confianza < 80%)" + $C.RESET)
    Write-FrameMid ("  " + $C.DIM + $C.SILVER + "14 tipos - 4149 cartas - GPU CUDA requerida" + $C.RESET)
    Write-Host ""
    try {
        $gpu = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
        if ($gpu) { Write-FrameMid ("  " + $C.GOLD + "GPU: " + $C.RESET + $C.SILVER + $($gpu[0]) + $C.RESET) }
    } catch {}
    Write-FrameBot
    Write-Host ""
    Write-DimGrim "  Presiona cualquier tecla para volver al grimorio..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 2>$null
}

function Invoke-Build {
    param([string]$Config = "RelWithDebInfo", [bool]$WithTesseract = $true)
    Clear-Host
    Write-FrameTop
    Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + "RITUAL DE CONSTRUCCION  " + $C.GOLD + "[" + $Config + "]" + $C.RESET)
    Write-FrameSep
    Write-Host ""
    Push-Location $SCRIPT:PROJECT_ROOT
    if (Test-Path $SCRIPT:BUILD_DIR) {
        Write-Info "Purgando build anterior..."
        Remove-Item -Recurse -Force $SCRIPT:BUILD_DIR
    }
    $cmakeArgs = @("--preset", "windows-x64", "-DUSE_SYSTEM_OPENCV=ON", "-DOpenCV_DIR=C:/opencv/build")
    if ($WithTesseract) {
        $cmakeArgs += "-DUSE_SYSTEM_TESSERACT=ON"
        Write-Success "Tesseract OCR: HABILITADO"
    } else {
        Write-Info "Tesseract OCR: deshabilitado"
    }
    Write-Info "Configurando CMake..."
    cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "Configuracion CMake FALLO"; Pop-Location; return $false }
    Write-Success "CMake configurado"
    Write-Info "Compilando ($Config)..."
    cmake --build --preset windows-x64 --config $Config --parallel
    if ($LASTEXITCODE -ne 0) { Write-Error "Compilacion FALLO"; Pop-Location; return $false }
    Write-Success "Compilacion completada"
    $releaseDir = "$SCRIPT:RELEASE_DIR/$Config"
    if (Test-Path $releaseDir) { Remove-Item -Recurse -Force $releaseDir }
    cmake --install $SCRIPT:BUILD_DIR --prefix $releaseDir --config $Config
    if ($LASTEXITCODE -ne 0) { Write-Error "Instalacion FALLO"; Pop-Location; return $false }
    Write-Success ("Plugin instalado en: " + $releaseDir)
    Pop-Location
    return $true
}

function Invoke-Deploy {
    param([string]$Config = "RelWithDebInfo")
    Clear-Host
    Write-FrameTop
    Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + "RITUAL DE INVOCACION - Desplegando en OBS" + $C.RESET)
    Write-FrameSep
    Write-Host ""
    $srcDll  = "$SCRIPT:RELEASE_DIR/$Config/obs-plugins/64bit/vtes-card-scanner.dll"
    $srcData = "$SCRIPT:RELEASE_DIR/$Config/data/obs-plugins/vtes-card-scanner"
    $dstDll  = $SCRIPT:OBS_PLUGIN_DIR
    $dstData = $SCRIPT:OBS_DATA_DIR
    $obsProc = Get-Process -Name "obs64" -ErrorAction SilentlyContinue
    if ($obsProc) { Write-Info "OBS esta ejecutandose - se cerrara..."; Stop-Process -Name "obs64" -Force; Start-Sleep -Seconds 2 }
    if (Test-Path $srcDll) {
        if (-not (Test-Path $dstDll)) { New-Item -ItemType Directory -Path $dstDll -Force | Out-Null }
        Copy-Item $srcDll "$dstDll\" -Force
        Write-Success ("DLL copiado: " + $srcDll + " -> " + $dstDll)
    } else { Write-Error "DLL no encontrado en: $srcDll"; Write-Info "Ejecuta BUILD primero"; return $false }
    $srcPdb = "$SCRIPT:RELEASE_DIR/$Config/obs-plugins/64bit/vtes-card-scanner.pdb"
    if (Test-Path $srcPdb) { Copy-Item $srcPdb "$dstDll\" -Force }
    if (Test-Path $srcData) {
        if (Test-Path $dstData) { Remove-Item -Recurse -Force $dstData -ErrorAction SilentlyContinue }
        Copy-Item $srcData $dstData -Recurse -Force
        Write-Success ("Data copiada: " + $srcData + " -> " + $dstData)
    } else { Write-Error "Data no encontrada en: $srcData" }
    $onnxDirs = @("$SCRIPT:BUILD_DIR/_deps/onnxruntime-src/lib", "$SCRIPT:BUILD_DIR/_deps/onnxruntime-src/bin")
    $copied = 0
    foreach ($dir in $onnxDirs) {
        if (Test-Path $dir) {
            Get-ChildItem "$dir\*.dll" -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne "DirectML.dll" } | ForEach-Object { Copy-Item $_.FullName "$dstDll\" -Force -ErrorAction SilentlyContinue; $copied++ }
        }
    }
    if ($copied -gt 0) { Write-Success ("ONNX Runtime DLLs: " + $copied + " copiadas") }
    # DirectML.dll ships with Windows 10 1903+ — only copy if system has it
    $sysDml = "$env:SystemRoot\System32\DirectML.dll"
    if (Test-Path $sysDml) { Copy-Item $sysDml "$dstDll\" -Force -ErrorAction SilentlyContinue; Write-Success "DirectML.dll: copiada desde System32" }
    $cvDll = Get-ChildItem "C:\opencv\build\x64\vc16\bin\opencv_world5*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $cvDll) { $cvDll = Get-ChildItem "C:\opencv\build\x64\vc16\bin\opencv_world4*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1 }
    if ($cvDll) { Copy-Item $cvDll.FullName "$dstDll\" -Force; Write-Success ("OpenCV DLL: " + $cvDll.Name) }
    Write-Host ""
    Write-Success "Invocacion completa. Abriendo OBS..."
    $obsExe = "C:\Program Files\obs-studio\bin\64bit\obs64.exe"
    if (Test-Path $obsExe) { Start-Process $obsExe }
    return $true
}

function Invoke-CopyPerType {
    Clear-Host
    Write-FrameTop
    Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + "RITUAL DE SINCRONIZACION - Per-Type Models" + $C.RESET)
    Write-FrameSep
    Write-Host ""
    $wslTest = bash -c "ls $SCRIPT:WSL_DATA_DIR/vtes_embedder_vampire.onnx 2>/dev/null && echo OK" 2>$null
    if ($wslTest -ne "OK") {
        Write-Error "WSL models no encontrados en: $SCRIPT:WSL_DATA_DIR"
        Write-Info "Entrena primero con: python scripts/train_per_type.py"
        return $false
    }
    Write-Info ("Origen WSL:  " + $SCRIPT:WSL_DATA_DIR)
    Write-Info ("Destino:     " + $SCRIPT:PER_TYPE_DIR)
    if (-not (Test-Path $SCRIPT:PER_TYPE_DIR)) { New-Item -ItemType Directory -Path $SCRIPT:PER_TYPE_DIR -Force | Out-Null }
    $patterns = @("vtes_embedder_*.onnx", "vtes_embedder_*.onnx.data", "embeddings_*.bin", "embeddings_*_meta.json", "per_type_manifest.json")
    $count = 0
    foreach ($pat in $patterns) {
        $files = bash -c "ls $SCRIPT:WSL_DATA_DIR/$pat 2>/dev/null" 2>$null
        if ($files) {
            foreach ($f in ($files -split "`n")) {
                $f = $f.Trim()
                if ([string]::IsNullOrWhiteSpace($f)) { continue }
                Write-Info ("Copiando: " + (Split-Path -Leaf $f))
                bash -c "cp \`"$f\`" ~/windows/C/.../" 2>$null
                $count++
            }
        }
    }
    $rootManifest = bash -c "ls $SCRIPT:WSL_MANIFEST 2>/dev/null && echo OK" 2>$null
    if ($rootManifest -eq "OK") { Write-Info "Manifest actualizado" }
    Write-Host ""
    Write-Success ("Sincronizacion completa: " + $count + " archivos copiados")
    Write-Info "Modelos listos para usarse en el plugin"
    return $true
}

function Invoke-InstallTesseract {
    Clear-Host
    Write-FrameTop
    Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + "RITUAL DE INVOCACION - Tesseract OCR" + $C.RESET)
    Write-FrameSep
    Write-Host ""
    if (Test-TesseractReady) {
        Write-Success ("Tesseract ya esta instalado en: " + $SCRIPT:TESSERACT_PATH)
        return $true
    }
    Write-Info "Instalando Tesseract OCR via winget..."
    winget install -e --id tesseract-ocr.tesseract --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Instalacion con winget FALLO"
        Write-Info "Descarga manual: https://github.com/UB-Mannheim/tesseract/releases"
        Write-Info "Asegurate de incluir English language data en la instalacion"
        return $false
    }
    Write-Success "Tesseract OCR instalado"
    Write-Info ("Verifica que eng.traineddata existe en: " + $SCRIPT:TESSERACT_PATH + "\tessdata\")
    return $true
}

$SCRIPT:NSIS_PATH = "C:\Program Files (x86)\NSIS"

function Test-NsisReady {
    $null = Get-Command "makensis" -ErrorAction SilentlyContinue
    if ($?) { return $true }
    $nsisExe = Join-Path $SCRIPT:NSIS_PATH "makensis.exe"
    return (Test-Path $nsisExe)
}

function Invoke-Package {
    param([string]$Config = "RelWithDebInfo")
    Clear-Host
    Write-FrameTop
    Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + "RITUAL DE EMPAQUETADO - NSIS Installer" + $C.RESET)
    Write-FrameSep
    Write-Host ""
    # ── 1. Find makensis ────────────────────────────────────────────
    $nsisExe = $null
    $null = Get-Command "makensis" -ErrorAction SilentlyContinue
    if ($?) { $nsisExe = (Get-Command "makensis").Source }
    else {
        $alt = Join-Path $SCRIPT:NSIS_PATH "makensis.exe"
        if (Test-Path $alt) { $nsisExe = $alt }
    }
    if (-not $nsisExe) {
        Write-Error "NSIS (makensis.exe) no encontrado."
        Write-Info "Busca en: $SCRIPT:NSIS_PATH"
        Write-Info "Descarga: https://nsis.sourceforge.io/Download"
        return $false
    }
    Write-Info "NSIS encontrado: $nsisExe"
    # ── 2. Verify build output exists ──────────────────────────────
    $releaseDir = Join-Path $SCRIPT:RELEASE_DIR $Config
    $dllPath = Join-Path $releaseDir "obs-plugins\64bit\vtes-card-scanner.dll"
    if (-not (Test-Path $dllPath)) {
        Write-Error "Build output no encontrado en: $releaseDir"
        Write-Info "Ejecuta BUILD primero"
        return $false
    }
    Write-Info "Build output: $releaseDir"
    # ── 3. Generate NSIS script ────────────────────────────────────
    $nsiPath = Join-Path $SCRIPT:PROJECT_ROOT "build_installer.nsi"
    $outFileName = "vtes-card-scanner-Installer-$Config-x64.exe"
    $outPath = Join-Path $SCRIPT:RELEASE_DIR $outFileName
    $nsiContent = @'
; VTES Card Scanner - NSIS Installer
SetCompressor lzma
!include "MUI2.nsh"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN_TEXT "Launch OBS Studio"
!define MUI_FINISHPAGE_RUN "$INSTDIR\bin\64bit\obs64.exe"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "__PROJECT_ROOT__\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"
Name "VTES Card Scanner"
OutFile "__OUT_PATH__"
InstallDir "$PROGRAMFILES64\obs-studio"
Section "Plugin" SEC_PLUGIN
  SectionIn RO
  SetOutPath "$INSTDIR\obs-plugins\64bit"
  File "__RELEASE_DIR__\obs-plugins\64bit\vtes-card-scanner.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\vtes-card-scanner.pdb"
  SetOutPath "$INSTDIR\data\obs-plugins\vtes-card-scanner"
  File /r "__RELEASE_DIR__\data\obs-plugins\vtes-card-scanner\*.*"
SectionEnd
Section "ONNX Runtime" SEC_ONNX
  SectionIn RO
  SetOutPath "$INSTDIR\obs-plugins\64bit"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\onnxruntime.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\onnxruntime_providers_shared.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\DirectML.dll"
SectionEnd
Section "OpenCV Runtime" SEC_OPENCV
  SectionIn RO
  SetOutPath "$INSTDIR\obs-plugins\64bit"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\opencv_world*.dll"
SectionEnd
Section -Post
  WriteUninstaller "$INSTDIR\obs-plugins\64bit\uninstall-vtes-card-scanner.exe"
  WriteRegStr HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VTES Card Scanner" "DisplayName" "VTES Card Scanner OBS Plugin"
  WriteRegStr HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VTES Card Scanner" "UninstallString" "$INSTDIR\obs-plugins\64bit\uninstall-vtes-card-scanner.exe"
  WriteRegStr HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VTES Card Scanner" "DisplayVersion" "__CONFIG__"
  WriteRegStr HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VTES Card Scanner" "Publisher" "VTES Card Scanner"
SectionEnd
Section Uninstall
  Delete "$INSTDIR\obs-plugins\64bit\vtes-card-scanner.dll"
  Delete "$INSTDIR\obs-plugins\64bit\vtes-card-scanner.pdb"
  Delete "$INSTDIR\obs-plugins\64bit\onnxruntime.dll"
  Delete "$INSTDIR\obs-plugins\64bit\onnxruntime_providers_shared.dll"
  Delete "$INSTDIR\obs-plugins\64bit\DirectML.dll"
  Delete "$INSTDIR\obs-plugins\64bit\opencv_world*.dll"
  RMDir /r "$INSTDIR\data\obs-plugins\vtes-card-scanner"
  Delete "$INSTDIR\obs-plugins\64bit\uninstall-vtes-card-scanner.exe"
  DeleteRegKey HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VTES Card Scanner"
SectionEnd
'@
    $nsiContent = $nsiContent.Replace("__RELEASE_DIR__", $releaseDir)
    $nsiContent = $nsiContent.Replace("__PROJECT_ROOT__", $SCRIPT:PROJECT_ROOT)
    $nsiContent = $nsiContent.Replace("__OUT_PATH__", $outPath)
    $nsiContent = $nsiContent.Replace("__CONFIG__", $Config)
    [System.IO.File]::WriteAllText($nsiPath, $nsiContent, [System.Text.Encoding]::UTF8)
    Write-Info "Script NSIS generado: $nsiPath"
    # ── 4. Compile installer ──────────────────────────────────────
    Push-Location $SCRIPT:PROJECT_ROOT
    Write-Info "Compilando instalador..."
    $outLog = Join-Path $SCRIPT:PROJECT_ROOT "build_installer_out.log"
    $errLog = Join-Path $SCRIPT:PROJECT_ROOT "build_installer_err.log"
    $proc = Start-Process -FilePath $nsisExe -ArgumentList "/V4", $nsiPath -NoNewWindow -RedirectStandardOutput $outLog -RedirectStandardError $errLog -Wait -PassThru
    $exitCode = $proc.ExitCode
    Pop-Location
    if (Test-Path $outLog) {
        $c = Get-Content $outLog -Raw
        if ($c -and $c.Trim()) { Write-Host ($C.DIM + $C.SILVER + $c + $C.RESET) }
        Remove-Item $outLog -ErrorAction SilentlyContinue
    }
    if (Test-Path $errLog) {
        $c = Get-Content $errLog -Raw
        if ($c -and $c.Trim()) { Write-Host ($C.DIM + $C.CRIMSON + $c + $C.RESET) }
        Remove-Item $errLog -ErrorAction SilentlyContinue
    }
    # Clean up temp .nsi
    Remove-Item $nsiPath -ErrorAction SilentlyContinue
    if ($exitCode -ne 0) {
        Write-Error "NSIS FALLO (exit code: $exitCode)"
        return $false
    }
    if (-not (Test-Path $outPath)) {
        Write-Error "Instalador no generado en: $outPath"
        return $false
    }
    $size = [math]::Round((Get-Item $outPath).Length / 1MB, 0)
    Write-Host ""
    Write-Success "Instalador generado:"
    Write-Info "  $outPath"
    Write-Info "  Tamaño: $size MB"
    Write-Host ""
    Write-DimGrim "  El instalador detecta OBS desde el registro de Windows"
    Write-DimGrim "  Por defecto: C:\Program Files\obs-studio"
    return $true
}

function Invoke-DeployAll {
    param([string]$Config = "RelWithDebInfo")
    Clear-Host
    Write-FrameTop
    Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + "RITUAL COMPLETO - Build + Deploy" + $C.RESET)
    Write-FrameSep
    Write-Host ""
    $tesseractAvailable = Test-TesseractReady
    Write-Banner "FASE 1: CONSTRUCCION"
    $ok = Invoke-Build -Config $Config -WithTesseract $tesseractAvailable
    if (-not $ok) { Write-Error "Construccion fallida. Abortando."; return }
    Write-Host ""
    Write-Banner "FASE 2: INVOCACION"
    $ok = Invoke-Deploy -Config $Config
    if (-not $ok) { Write-Error "Deploy fallido."; return }
    Write-Host ""
    Write-FrameSep
    Write-FrameMid ("  " + $C.BOLD + $C.GOLD + "RITUAL COMPLETADO CON EXITO" + $C.RESET)
    Write-FrameBot
    Start-Sleep -Seconds 2
}

function Show-Menu {
    $running = $true
    do {
        Clear-Host
        Show-Ankh
        Write-Host ""
        Show-Header
        $buildOK  = Test-BuildReady
        $modelsOK = Test-ModelsReady
        $tessOK   = Test-TesseractReady
        Update-StatusLine -BuildStatus $buildOK -ModelsStatus $modelsOK -TesseractStatus $tessOK
        Write-FrameTop
        Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + "EL GRIMORIO - Escoge tu ritual" + $C.RESET)
        Write-FrameSep
        Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + " 1" + $C.RESET + "  " + $C.BOLD + "BUILD PLUGIN" + $C.RESET + "       " + $C.DIM + $C.SILVER + "Compila el plugin OBS" + $C.RESET)
        Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + " 2" + $C.RESET + "  " + $C.BOLD + "DEPLOY TO OBS" + $C.RESET + "     " + $C.DIM + $C.SILVER + "Copia DLL + data a OBS" + $C.RESET)
        Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + " 3" + $C.RESET + "  " + $C.BOLD + "COPY PER-TYPE" + $C.RESET + "     " + $C.DIM + $C.SILVER + "WSL -> plugin (modelos por tipo)" + $C.RESET)
        Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + " 4" + $C.RESET + "  " + $C.BOLD + "INSTALL TESSERACT" + $C.RESET + " " + $C.DIM + $C.SILVER + "winget -> OCR engine" + $C.RESET)
        Write-FrameSep
        Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + " 5" + $C.RESET + "  " + $C.BOLD + "VERIFY STATUS" + $C.RESET + "     " + $C.DIM + $C.SILVER + "Diagnostico completo" + $C.RESET)
        Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + " 6" + $C.RESET + "  " + $C.BOLD + "DEPLOY ALL" + $C.RESET + "        " + $C.DIM + $C.SILVER + "Build + Deploy en un solo paso" + $C.RESET)
        Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + " 7" + $C.RESET + "  " + $C.BOLD + "PACKAGE INSTALLER" + $C.RESET + " " + $C.DIM + $C.SILVER + "Genera .exe (NSIS) para distribuir" + $C.RESET)
        Write-FrameSep
        Write-FrameMid ("  " + $C.DIM + $C.SILVER + " Q" + $C.RESET + "  " + $C.RESET + "QUIT" + $C.RESET + "              " + $C.DIM + $C.SILVER + "Abandonar el grimorio" + $C.RESET)
        Write-FrameBot
        Write-Host ""
        Write-Host ($C.GOLD + "> " + $C.RESET) -NoNewline
        $choice = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown").Character
        Write-Host ($C.GOLD + $choice + $C.RESET)
        switch ($choice) {
            "1" { Invoke-Build -WithTesseract (Test-TesseractReady); Write-Host ""; Write-DimGrim "  Presiona cualquier tecla..."; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 2>$null }
            "2" { Invoke-Deploy; Write-Host ""; Write-DimGrim "  Presiona cualquier tecla..."; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 2>$null }
            "3" { Invoke-CopyPerType; Write-Host ""; Write-DimGrim "  Presiona cualquier tecla..."; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 2>$null }
            "4" { Invoke-InstallTesseract; Write-Host ""; Write-DimGrim "  Presiona cualquier tecla..."; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 2>$null }
            "5" { Invoke-Verify }
            "6" { Invoke-DeployAll; Write-Host ""; Write-DimGrim "  Presiona cualquier tecla..."; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 2>$null }
            "7" { Invoke-Package; Write-Host ""; Write-DimGrim "  Presiona cualquier tecla..."; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 2>$null }
            "q" { $running = $false }
            "Q" { $running = $false }
        }
    } while ($running)
    Clear-Host
    Show-Ankh
    Write-Host ""
    Write-FrameTop
    Write-FrameMid ("  " + $C.DIM + $C.SILVER + "El grimorio se cierra. La sed de sangre" + $C.RESET)
    Write-FrameMid ("  " + $C.DIM + $C.SILVER + "esperara otra noche..." + $C.RESET)
    Write-FrameBot
    Write-Host ""
    Write-DimGrim "Hasta el proximo ritual, Cainita."
    Start-Sleep -Seconds 2
}

function Main {
    if ($Host.UI.RawUI) {
        $Host.UI.RawUI.ForegroundColor = "White"
        $Host.UI.RawUI.BackgroundColor = "Black"
    }
    $cmdLine = $MyInvocation.Line.Trim()
    if ($cmdLine -match "\S+\.ps1\s+(\w+)") {
        $action = $Matches[1].ToLower()
        switch ($action) {
            "build"     { Invoke-Build -WithTesseract (Test-TesseractReady); return }
            "deploy"    { Invoke-Deploy; return }
            "verify"    { Invoke-Verify; return }
            "sync"      { Invoke-CopyPerType; return }
            "tesseract" { Invoke-InstallTesseract; return }
            "all"       { Invoke-DeployAll; return }
            "package"   { Invoke-Package; return }
        }
    }
    Show-Menu
}

Main