param(
    [string]$Background = "paper2/paper2-1/background_0_5_w001_1h.npz",
    [string]$Config = "paper2/paper2-1/nearfield_starlink_multiwindow_config.yaml",
    [string]$OutputRoot = "paper2/paper2-1/topn_sensitivity_w001",
    [int[]]$TopN = @(10, 25, 50, 100),
    [int]$MaxScanSatellites = 5000,
    [int]$PhaseNullTrials = 500,
    [string]$Pol = "ee",
    [string]$PolarizationMode = "jones_anti_correlated",
    [double]$PolarizationContrast = 0.35
)

$ErrorActionPreference = "Stop"

$wslRepo = "/mnt/c/Users/kimeu/Downloads/starlink_uemr-main/starlink_uemr-main"
$python = "/home/kimeuhanyoon18/miniconda3/envs/starlink_env/bin/python"
$script = "paper2/paper2-1/run_nearfield_starlink_multiwindow_v4.py"

foreach ($n in $TopN) {
    $out = "$OutputRoot/top$n"
    $cmd = "cd $wslRepo && $python $script --config $Config --background $Background --output-dir $out --start-window 0 --max-windows 1 --max-sats-per-window $n --max-scan-satellites $MaxScanSatellites --peak-alt-min-deg 35 --peak-alt-max-deg 70 --cap-selection-method predicted_peak_apparent_flux --pol $Pol --polarization-mode $PolarizationMode --polarization-contrast $PolarizationContrast --phase-null-trials $PhaseNullTrials"
    wsl bash -lc $cmd
}

$repoRoot = "C:\Users\kimeu\Downloads\starlink_uemr-main\starlink_uemr-main"
$rows = Get-ChildItem -Path (Join-Path $repoRoot $OutputRoot) -Recurse -Filter multi_window_summary.csv |
    ForEach-Object {
        $case = Split-Path (Split-Path $_.FullName -Parent) -Parent | Split-Path -Leaf
        Import-Csv $_.FullName | ForEach-Object {
            $_ | Add-Member -NotePropertyName case -NotePropertyValue $case -PassThru
        }
    }

$summaryPath = Join-Path $repoRoot "$OutputRoot\topn_summary.csv"
New-Item -ItemType Directory -Force -Path (Split-Path $summaryPath -Parent) | Out-Null
$rows | Export-Csv -NoTypeInformation -Path $summaryPath
Write-Host "Wrote $summaryPath"
