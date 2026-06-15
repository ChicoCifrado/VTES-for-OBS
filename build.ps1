<#
.SYNOPSIS
    Build VTES OBS plugin (wrapper for build-windows.ps1)
#>

param(
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Configuration = 'Release',
    
    [switch]$Clean,
    [switch]$BuildInstaller
)

$ProjectRoot = $PSScriptRoot
$BuildScript = Join-Path $ProjectRoot 'build-windows.ps1'

if (-not (Test-Path $BuildScript)) {
    Write-Error "build-windows.ps1 not found"
    exit 1
}

$splat = @{ Configuration = $Configuration }
if ($Clean) { $splat['Clean'] = $true }
if ($BuildInstaller) { $splat['BuildInstaller'] = $true }

& $BuildScript @splat
exit $LASTEXITCODE