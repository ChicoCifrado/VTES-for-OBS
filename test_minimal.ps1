#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Minimal test
#>
#requires -Version 5.1
$ErrorActionPreference = 'Stop'

function Write-FrameTop {
    Write-Host "top"
}
function Write-FrameMid {
    Write-Host "mid"
}
function Write-FrameBot {
    Write-Host "bot"
}

function Invoke-Build {
    param([bool]$WithTesseract = $true)
    Write-Host "build"
}
function Invoke-Deploy {
    Write-Host "deploy"
}
function Invoke-Verify {
    Write-Host "verify"
}
function Invoke-CopyPerType {
    Write-Host "sync"
}
function Invoke-InstallTesseract {
    Write-Host "tesseract"
}
function Invoke-DeployAll {
    Write-Host "all"
}
function Show-Menu {
    Write-Host "menu"
}

function Main {
    $args = $MyInvocation.Line.Trim()
    if ($args -match '^\S+\.ps1\s+(\w+)') {
        $action = $Matches[1].ToLower()
        switch ($action) {
            'build'     { Invoke-Build -WithTesseract $true; return }
            'deploy'    { Invoke-Deploy; return }
            'verify'    { Invoke-Verify; return }
            'sync'      { Invoke-CopyPerType; return }
            'tesseract' { Invoke-InstallTesseract; return }
            'all'       { Invoke-DeployAll; return }
        }
    }
    Show-Menu
}
Main
