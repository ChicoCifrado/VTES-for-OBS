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
    return Test-TesseractReady
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
    Write-FrameMid ("  " + $C.DIM + $C.SILVER + "Pipeline: YOLO(Ort::Session) -> Type(Ort::Session) -> Per-Type(cv::dnn) -> Global" + $C.RESET)
    Write-FrameMid ("  " + $C.DIM + $C.SILVER + "          -> OCR Fallback (cuando confianza < 80%)" + $C.RESET)
    Write-FrameMid ("  " + $C.DIM + $C.SILVER + "14 tipos - 4149 cartas - GPU CUDA (ORT CUDA provider)" + $C.RESET)
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
    param([string]$Config = "RelWithDebInfo")
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
    $cmakeArgs = @("--preset", "windows-x64")
    $tesseractAvail = Test-TesseractReady
    if ($tesseractAvail) {
        Write-Success "Tesseract OCR runtime disponible (DLL cargada dinamicamente)"
    } else {
        Write-Info "Tesseract OCR runtime NO disponible"
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
    # ── Copy CUDA runtime DLLs to release dir (for NSIS installer) ─
    $cudaPaths = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    )
    $cudaDlls = @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
                  "cufft64_11.dll")
    $cudaCopied = 0
    $releasePluginDir = "$releaseDir/obs-plugins/64bit"
    foreach ($cudaPath in $cudaPaths) {
        if (-not (Test-Path $cudaPath)) { continue }
        foreach ($dll in $cudaDlls) {
            $src = Join-Path $cudaPath $dll
            if (Test-Path $src) {
                Copy-Item $src "$releasePluginDir\" -Force -ErrorAction SilentlyContinue
                $cudaCopied++
            }
        }
    }
    if ($cudaCopied -gt 0) { Write-Info ("CUDA runtime DLLs: " + $cudaCopied + " copiadas a release") }
    # ── Copy TensorRT DLLs to release dir (if TENSORRT_ROOT is set) ─
    $trtRoot = $env:TENSORRT_ROOT
    if ($trtRoot) {
        $trtDlls = @("nvinfer_lean_11.dll", "nvinfer_plugin_11.dll")
        $trtCopied = 0
        foreach ($trtDir in @("$trtRoot\bin", "$trtRoot\lib")) {
            if (-not (Test-Path $trtDir)) { continue }
            foreach ($dll in $trtDlls) {
                $src = Join-Path $trtDir $dll
                if (Test-Path $src) {
                    Copy-Item $src "$releasePluginDir\" -Force -ErrorAction SilentlyContinue
                    $trtCopied++
                }
            }
        }
        if ($trtCopied -gt 0) { Write-Info ("TensorRT DLLs: " + $trtCopied + " copiadas a release") }
    }
    # ── Copy TensorRT engine file to release data models dir ─
    $engineFile = Join-Path $SCRIPT:PROJECT_ROOT "vtes.engine"
    $modelsDir = "$releaseDir/data/obs-plugins/vtes-card-scanner/models"
    if (Test-Path $engineFile) {
        if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null }
        Copy-Item $engineFile "$modelsDir\" -Force
        Write-Success "TensorRT engine copiado a release: $modelsDir"
    } else {
        Write-Error "TensorRT engine NO encontrado en: $engineFile"
        Write-Info "Generalo con: trtexec --onnx=data/models/vtes.onnx --saveEngine=vtes.engine --versionCompatible"
    }
    # ── cudnn64_9.dll ya no se copia — TensorRT no requiere cuDNN ─
    Pop-Location
    return $true
}

function Invoke-Clean {
    Clear-Host
    Write-FrameTop
    Write-FrameMid ("  " + $C.BOLD + $C.CRIMSON + "RITUAL DE PURIFICACION - Limpiando build_x64" + $C.RESET)
    Write-FrameSep
    Write-Host ""
    Push-Location $SCRIPT:PROJECT_ROOT
    if (Test-Path $SCRIPT:BUILD_DIR) {
        Write-Info "Purgando $($SCRIPT:BUILD_DIR)..."
        Remove-Item -Recurse -Force $SCRIPT:BUILD_DIR
        Write-Success "build_x64 eliminado"
    } else {
        Write-Info "build_x64 no existe - nada que limpiar"
    }
    Pop-Location
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
    $srcPluginDir = "$SCRIPT:RELEASE_DIR/$Config/obs-plugins/64bit"
    $dstDll  = $SCRIPT:OBS_PLUGIN_DIR
    $dstData = $SCRIPT:OBS_DATA_DIR
    $obsProc = Get-Process -Name "obs64" -ErrorAction SilentlyContinue
    if ($obsProc) { Write-Info "OBS esta ejecutandose - se cerrara..."; Stop-Process -Name "obs64" -Force; Start-Sleep -Seconds 2 }
    if (-not (Test-Path $srcDll)) { Write-Error "DLL no encontrado en: $srcDll"; Write-Info "Ejecuta BUILD primero"; return $false }
    if (-not (Test-Path $dstDll)) { New-Item -ItemType Directory -Path $dstDll -Force | Out-Null }
    # ── Copy all DLLs from release dir (plugin + ORT + OpenCV) ─
    Get-ChildItem "$srcPluginDir\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName "$dstDll\" -Force -ErrorAction SilentlyContinue
    }
    Write-Success ("DLLs copiados: " + (Get-ChildItem "$srcPluginDir\*.dll" | Measure-Object | Select-Object -ExpandProperty Count) + " archivos")
    $srcPdb = "$SCRIPT:RELEASE_DIR/$Config/obs-plugins/64bit/vtes-card-scanner.pdb"
    if (Test-Path $srcPdb) { Copy-Item $srcPdb "$dstDll\" -Force }
    if (Test-Path $srcData) {
        if (Test-Path $dstData) { Remove-Item -Recurse -Force $dstData -ErrorAction SilentlyContinue }
        Copy-Item $srcData $dstData -Recurse -Force
        Write-Success ("Data copiada: " + $srcData + " -> " + $dstData)
    } else { Write-Error "Data no encontrada en: $srcData" }
    # ── Fallback: ensure vtes.engine is in OBS data models dir ─────
    $engineSrc = Join-Path $SCRIPT:PROJECT_ROOT "vtes.engine"
    $engineDstDir = Join-Path $dstData "models"
    $engineDst = Join-Path $engineDstDir "vtes.engine"
    if (Test-Path $engineSrc) {
        if (-not (Test-Path $engineDstDir)) { New-Item -ItemType Directory -Path $engineDstDir -Force | Out-Null }
        Copy-Item $engineSrc $engineDst -Force
        Write-Success "TensorRT engine asegurado en: $engineDst"
    }
    # ── CUDA Runtime DLLs ─
    $cudaPaths = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    )
    $cudaDlls = @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
                  "cufft64_11.dll")
    $cudaCopied = 0
    foreach ($cudaPath in $cudaPaths) {
        if (-not (Test-Path $cudaPath)) { continue }
        foreach ($dll in $cudaDlls) {
            $src = Join-Path $cudaPath $dll
            if (Test-Path $src) {
                Copy-Item $src "$dstDll\" -Force -ErrorAction SilentlyContinue
                $cudaCopied++
            }
        }
    }
    if ($cudaCopied -gt 0) {
        Write-Success ("CUDA runtime DLLs: " + $cudaCopied + " copiadas a obs-plugins/64bit")
    }
    # ── TensorRT DLLs (desde TENSORRT_ROOT) ─
    $trtRoot = $env:TENSORRT_ROOT
    if ($trtRoot) {
        $trtDlls = @("nvinfer_lean_11.dll", "nvinfer_plugin_11.dll")
        $trtCopied = 0
        foreach ($trtDir in @("$trtRoot\bin", "$trtRoot\lib")) {
            if (-not (Test-Path $trtDir)) { continue }
            foreach ($dll in $trtDlls) {
                $src = Join-Path $trtDir $dll
                if (Test-Path $src) {
                    Copy-Item $src "$dstDll\" -Force -ErrorAction SilentlyContinue
                    $trtCopied++
                }
            }
        }
        if ($trtCopied -gt 0) {
            Write-Info ("TensorRT DLLs: " + $trtCopied + " copiadas a obs-plugins/64bit")
            # Also copy to OBS bin/64bit/
            $obsBin = "C:\Program Files\obs-studio\bin\64bit"
            if (Test-Path $obsBin) {
                foreach ($trtDir in @("$trtRoot\bin", "$trtRoot\lib")) {
                    if (-not (Test-Path $trtDir)) { continue }
                    foreach ($dll in $trtDlls) {
                        $src = Join-Path $trtDir $dll
                        if (Test-Path $src) {
                            Copy-Item $src "$obsBin\" -Force -ErrorAction SilentlyContinue
                        }
                    }
                }
                Write-Info "TensorRT DLLs tambien copiadas a $obsBin"
            }
        }
    }
    # DirectML.dll ships with Windows 10 1903+ — copy as DirectML fallback
    $sysDml = "$env:SystemRoot\System32\DirectML.dll"
    if (Test-Path $sysDml) { Copy-Item $sysDml "$dstDll\" -Force -ErrorAction SilentlyContinue; Write-Info "DirectML.dll: disponible como fallback" }
    # OpenCV DLLs ya instalados por cmake --install en release dir
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
                & wsl cp "$f" "$(wslpath "$SCRIPT:PER_TYPE_DIR")/" 2>$null
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
    $tesseractReady = Test-TesseractReady
    if ($tesseractReady) {
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
RequestExecutionLevel admin
!include "MUI2.nsh"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_NOREBOOTSUPPORT
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
InstallDirRegKey HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\obs-studio" "InstallLocation"
Section "Plugin" SEC_PLUGIN
  SectionIn RO
  SetOutPath "$INSTDIR\obs-plugins\64bit"
  File "__RELEASE_DIR__\obs-plugins\64bit\vtes-card-scanner.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\vtes-card-scanner.pdb"
  ; 3rd-party DLLs (all present in release dir after cmake --install)
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\onnxruntime.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\onnxruntime_providers_shared.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\onnxruntime_providers_cuda.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\cudart64_12.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\cublas64_12.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\cublasLt64_12.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\cufft64_11.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\cudnn64_9.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\nvinfer_lean_11.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\nvinfer_plugin_11.dll"
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\DirectML.dll"
  ; OpenCV world DLL (version varies: opencv_world4xx.dll or opencv_world5xx.dll)
  File /nonfatal "__RELEASE_DIR__\obs-plugins\64bit\opencv_world*.dll"
  SetOutPath "$INSTDIR\data\obs-plugins\vtes-card-scanner"
  File /r "__RELEASE_DIR__\data\obs-plugins\vtes-card-scanner\*.*"
  ; TensorRT engine file → data/models/
  SetOutPath "$INSTDIR\data\obs-plugins\vtes-card-scanner\models"
  File /nonfatal "__PROJECT_ROOT__\vtes.engine"
SectionEnd
Section -Post
  WriteUninstaller "$INSTDIR\obs-plugins\64bit\uninstall-vtes-card-scanner.exe"
  WriteRegStr HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VTES Card Scanner" "DisplayName" "VTES Card Scanner OBS Plugin"
  WriteRegStr HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VTES Card Scanner" "UninstallString" "$INSTDIR\obs-plugins\64bit\uninstall-vtes-card-scanner.exe"
  WriteRegStr HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VTES Card Scanner" "DisplayVersion" "__CONFIG__"
  WriteRegStr HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VTES Card Scanner" "Publisher" "VTES Card Scanner"
  WriteRegStr HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VTES Card Scanner" "InstallLocation" "$INSTDIR"
SectionEnd
Section Uninstall
  SetOutPath "$INSTDIR"
  ; ── Clean OBS data dir (models, embeddings, logs, etc.) ──────────
  RMDir /r "$INSTDIR\data\obs-plugins\vtes-card-scanner"
  ; ── Remove VTES plugin DLLs ──────────────────────────────────────
  Delete "$INSTDIR\obs-plugins\64bit\vtes-card-scanner.dll"
  Delete "$INSTDIR\obs-plugins\64bit\vtes-card-scanner.pdb"
  ; ── Remove 3rd-party runtime DLLs (only ours — nonfatal if shared) ─
  Delete "$INSTDIR\obs-plugins\64bit\onnxruntime.dll"
  Delete "$INSTDIR\obs-plugins\64bit\onnxruntime_providers_shared.dll"
  Delete "$INSTDIR\obs-plugins\64bit\onnxruntime_providers_cuda.dll"
  Delete "$INSTDIR\obs-plugins\64bit\cudart64_12.dll"
  Delete "$INSTDIR\obs-plugins\64bit\cublas64_12.dll"
  Delete "$INSTDIR\obs-plugins\64bit\cublasLt64_12.dll"
  Delete "$INSTDIR\obs-plugins\64bit\cufft64_11.dll"
  Delete "$INSTDIR\obs-plugins\64bit\cudnn64_9.dll"
  Delete "$INSTDIR\obs-plugins\64bit\nvinfer_lean_11.dll"
  Delete "$INSTDIR\obs-plugins\64bit\nvinfer_plugin_11.dll"
  Delete "$INSTDIR\obs-plugins\64bit\DirectML.dll"
  Delete "$INSTDIR\obs-plugins\64bit\opencv_world*.dll"
  ; ── Retry locked files (OBS running) ──────────────────────────────
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\vtes-card-scanner.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\vtes-card-scanner.pdb"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\onnxruntime.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\onnxruntime_providers_shared.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\onnxruntime_providers_cuda.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\cudart64_12.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\cublas64_12.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\cublasLt64_12.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\cufft64_11.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\cudnn64_9.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\nvinfer_lean_11.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\nvinfer_plugin_11.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\DirectML.dll"
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\opencv_world*.dll"
  ; ── Remove empty directories ──────────────────────────────────────
  RMDir "$INSTDIR\obs-plugins\64bit"
  RMDir "$INSTDIR\obs-plugins"
  RMDir "$INSTDIR\data\obs-plugins\vtes-card-scanner"
  RMDir "$INSTDIR\data\obs-plugins"
  ; ── Uninstaller itself LAST ────────────────────────────────────────
  Delete /REBOOTOK "$INSTDIR\obs-plugins\64bit\uninstall-vtes-card-scanner.exe"
  ; ── Clean Windows registry ────────────────────────────────────────
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
    Write-Banner "FASE 1: CONSTRUCCION"
    $ok = Invoke-Build -Config $Config
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
            "1" { Invoke-Build; Write-Host ""; Write-DimGrim "  Presiona cualquier tecla..."; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 2>$null }
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
    if ($args.Count -gt 0) {
        $action = $args[0].ToLower()
        $cfg = if ($args.Count -gt 1) { $args[1] } else { "RelWithDebInfo" }
        switch ($action) {
            "build"     { Invoke-Build -Config $cfg; return }
            "deploy"    { Invoke-Deploy; return }
            "verify"    { Invoke-Verify; return }
            "sync"      { Invoke-CopyPerType; return }
            "tesseract" { Invoke-InstallTesseract; return }
            "all"       { Invoke-DeployAll; return }
            "package"   { Invoke-Package -Config $cfg; return }
            "installer" { Invoke-Package -Config $cfg; return }
            "clean"     { Invoke-Clean; return }
        }
    }
    Show-Menu
}

Main $args