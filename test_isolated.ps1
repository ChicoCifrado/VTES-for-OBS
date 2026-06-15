#!/usr/bin/env pwsh
#requires -Version 5.1

function Write-FrameTop { Write-Host "top" }
function Invoke-DeployAll { Write-Host "all" }

function Main {
    $args = $MyInvocation.Line.Trim()
    if ($args -match '\S+\.ps1\s+(\w+)') {
        $action = $Matches[1].ToLower()
        switch ($action) {
            'build'  { Invoke-DeployAll; return }
            'all'    { Invoke-DeployAll; return }
        }
    }
}
Main
